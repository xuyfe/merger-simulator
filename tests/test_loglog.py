"""Tests for LogLogDemand model."""
import numpy as np
import pandas as pd
import pytest

from merger_simulator.data import MergerData
from merger_simulator.demand.log_log import LogLogDemand


@pytest.fixture
def loglog_setup():
    """Create synthetic wide-format data for 3 firms across 100 markets."""
    rng = np.random.default_rng(456)
    n = 100

    B_true = np.array([
        [-2.5, 0.3, 0.2],
        [0.2, -3.0, 0.4],
        [0.1, 0.3, -2.0],
    ], dtype=float)

    alpha_true = np.array([8.0, 7.5, 7.0])
    mc_true = np.array([30.0, 25.0, 35.0])

    rows = []
    for m in range(n):
        cost_shock = rng.normal(0, 2, 3)
        mc = mc_true + cost_shock
        markup_frac = rng.uniform(0.1, 0.3, 3)
        p = mc * (1 + markup_frac)
        xi = rng.normal(0, 0.1, 3)
        logq = alpha_true + B_true @ np.log(p) + xi
        q = np.exp(logq)
        quarter = m % 4
        rows.append({
            "market_id": m, "quarter_id": quarter,
            "price_A": p[0], "price_B": p[1], "price_C": p[2],
            "passengers_A": q[0], "passengers_B": q[1], "passengers_C": q[2],
        })

    df = pd.DataFrame(rows)
    df["const"] = 1

    firms = ["A", "B", "C"]
    price_cols = {f: f"price_{f}" for f in firms}
    quantity_cols = {f: f"passengers_{f}" for f in firms}

    data = MergerData(
        df, firm_col="market_id", market_col="market_id",
        price_col="price_A", quantity_col="passengers_A",
    )

    for f in firms:
        h_sum = df.groupby("quarter_id")[price_cols[f]].transform("sum")
        h_n = df.groupby("quarter_id")[price_cols[f]].transform("count")
        df[f"iv_{f}"] = (h_sum - df[price_cols[f]]) / (h_n - 1)

    instruments = df[[f"iv_{f}" for f in firms]]

    model = LogLogDemand(firms=firms, price_cols=price_cols, quantity_cols=quantity_cols)
    return model, data, df, instruments, firms, price_cols, quantity_cols


class TestLogLogDemand:
    def test_estimation_runs(self, loglog_setup):
        model, data, df, instruments, firms, _, _ = loglog_setup
        model.estimate(data, instruments, exog_cols=["const"])
        assert model.B is not None
        assert model.B.shape == (3, 3)
        for i in range(3):
            assert model.B[i, i] < 0

    def test_elasticities_are_B(self, loglog_setup):
        """In log-log demand, B coefficients are the elasticities directly."""
        model, data, df, instruments, _, _, _ = loglog_setup
        model.estimate(data, instruments, exog_cols=["const"])
        for i in range(3):
            assert model.B[i, i] < -1, "Own-price elasticity should be elastic"

    def test_mc_recovery(self, loglog_setup):
        model, data, df, instruments, firms, price_cols, _ = loglog_setup
        model.estimate(data, instruments, exog_cols=["const"])
        mc = model.recover_marginal_costs(df)
        for f in firms:
            assert (mc[f] > 0).all()
            assert mc[f].mean() < df[price_cols[f]].mean()

    def test_merger_simulation(self, loglog_setup):
        model, data, df, instruments, firms, _, _ = loglog_setup
        model.estimate(data, instruments, exog_cols=["const"])
        result = model.simulate_merger(df, merging=("A", "B"), tol=0.1, damping=0.05)
        assert result["converged"] or result["diff"] < 1.0
        assert "A" in result["price_changes"]
        assert "B" in result["price_changes"]

    def test_predict_quantities(self, loglog_setup):
        model, data, df, instruments, firms, _, quantity_cols = loglog_setup
        model.estimate(data, instruments, exog_cols=["const"])
        q = model.predict_quantities(df.copy())
        for f in firms:
            assert (q[f] > 0).all()
            np.testing.assert_allclose(
                q[f], df[quantity_cols[f]].values, rtol=0.1,
            )
