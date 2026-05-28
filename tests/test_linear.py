"""Tests for LinearDemand model."""
import numpy as np
import pandas as pd
import pytest

from merger_simulator.data import MergerData
from merger_simulator.demand.linear import LinearDemand


@pytest.fixture
def linear_setup():
    """Create synthetic wide-format data for 3 products across 100 markets."""
    rng = np.random.default_rng(123)
    n = 100

    B_true = np.array([
        [-100, 20, 10],
        [15, -120, 25],
        [10, 20, -90],
    ], dtype=float)

    alpha_true = np.array([500, 450, 400], dtype=float)
    mc_true = np.array([3.0, 2.5, 3.5])

    rows = []
    for m in range(n):
        cost_shock = rng.normal(0, 0.3, 3)
        mc = mc_true + cost_shock
        markup = rng.uniform(0.3, 0.8, 3)
        p = mc + markup
        xi = rng.normal(0, 5, 3)
        q = alpha_true + B_true @ p + xi
        q = np.maximum(q, 1)
        city = m % 5
        quarter = m % 4
        rows.append({
            "market_id": m, "city_id": city, "quarter_id": quarter,
            "price1": p[0], "price2": p[1], "price3": p[2],
            "qty1": q[0], "qty2": q[1], "qty3": q[2],
        })

    df = pd.DataFrame(rows)
    df["const"] = 1

    firms = ["1", "2", "3"]
    price_cols = {f: f"price{f}" for f in firms}
    quantity_cols = {f: f"qty{f}" for f in firms}

    data = MergerData(
        df, firm_col="market_id", market_col="market_id",
        price_col="price1", quantity_col="qty1",
    )

    model = LinearDemand(firms=firms, price_cols=price_cols, quantity_cols=quantity_cols)

    for f in firms:
        h_sum = df.groupby("quarter_id")[price_cols[f]].transform("sum")
        h_n = df.groupby("quarter_id")[price_cols[f]].transform("count")
        df[f"iv_{f}"] = (h_sum - df[price_cols[f]]) / (h_n - 1)

    instruments = df[[f"iv_{f}" for f in firms]]

    return model, data, df, instruments, firms, price_cols, quantity_cols


class TestLinearDemand:
    def test_estimation_runs(self, linear_setup):
        model, data, df, instruments, firms, _, _ = linear_setup
        result = model.estimate(
            data, instruments,
            endog_cols=[f"price{f}" for f in firms],
            exog_cols=["const"],
        )
        assert model.B is not None
        assert model.B.shape == (3, 3)
        assert all(model.B[i, i] < 0 for i in range(3))

    def test_own_price_negative(self, linear_setup):
        model, data, df, instruments, firms, _, _ = linear_setup
        model.estimate(data, instruments,
                       endog_cols=[f"price{f}" for f in firms], exog_cols=["const"])
        for i in range(3):
            assert model.B[i, i] < 0, f"Own-price for firm {i} should be negative"

    def test_mc_recovery(self, linear_setup):
        model, data, df, instruments, firms, price_cols, quantity_cols = linear_setup
        model.estimate(data, instruments,
                       endog_cols=[f"price{f}" for f in firms], exog_cols=["const"])
        mc = model.recover_marginal_costs(df)
        for f in firms:
            assert mc[f].mean() < df[price_cols[f]].mean(), "Mean MC should be below mean price"
            assert np.isfinite(mc[f]).all(), f"MC for firm {f} should be finite"

    def test_merger_simulation(self, linear_setup):
        model, data, df, instruments, firms, price_cols, _ = linear_setup
        model.estimate(data, instruments,
                       endog_cols=[f"price{f}" for f in firms], exog_cols=["const"])
        result = model.simulate_merger(df, merging=("1", "2"), tol=0.01)
        assert result["converged"]
        assert "1" in result["price_changes"]
        assert "2" in result["price_changes"]

    def test_jacobian_is_B(self, linear_setup):
        model, data, df, instruments, firms, _, _ = linear_setup
        model.estimate(data, instruments,
                       endog_cols=[f"price{f}" for f in firms], exog_cols=["const"])
        jac = model.jacobian(data, market_id=0)
        np.testing.assert_array_equal(jac, model.B)
