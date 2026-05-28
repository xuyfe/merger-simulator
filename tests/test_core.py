"""Smoke tests for core package functionality using synthetic data."""
import numpy as np
import pandas as pd
import pytest

from merger_simulator.data import MergerData
from merger_simulator.demand.logit import Logit
from merger_simulator.demand.nested_logit import NestedLogit
from merger_simulator.merger import ownership_matrix


def make_synthetic_data(n_markets: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic merger-simulation data with 3 firms."""
    rng = np.random.default_rng(seed)
    rows = []
    firms = ["A", "B", "C"]
    for m in range(1, n_markets + 1):
        mkt_size = rng.integers(1000, 10000)
        xi = rng.normal(0, 0.1, size=3)
        cost_shifter = rng.normal(0, 1, size=3)
        mc = 2.0 + 0.2 * cost_shifter + rng.normal(0, 0.05, size=3)
        alpha = 2.0
        delta = np.array([1.5, 1.4, 0.8]) + xi
        price = mc + rng.uniform(0.1, 0.5, size=3)
        utility = delta - alpha * price
        exp_u = np.exp(utility)
        shares = exp_u / (1 + exp_u.sum())
        for i, f in enumerate(firms):
            rows.append({
                "firm": f, "market_id": m, "price": price[i],
                "share": shares[i], "market_size": mkt_size,
                "cost_shifter": cost_shifter[i], "advertising": rng.integers(0, 2),
                "city_id": (m % 5) + 1, "quarter_id": (m % 4) + 1,
            })
    return pd.DataFrame(rows)


class TestMergerData:
    def test_creation(self):
        df = make_synthetic_data()
        data = MergerData(
            df, firm_col="firm", market_col="market_id",
            price_col="price", share_col="share",
            market_size_col="market_size",
            cost_shifter_cols=["cost_shifter"],
        )
        assert data.n_markets == 50
        assert set(data.firms) == {"A", "B", "C"}

    def test_outside_shares(self):
        df = make_synthetic_data()
        data = MergerData(
            df, firm_col="firm", market_col="market_id",
            price_col="price", share_col="share",
        )
        s0 = data.outside_shares()
        assert (s0 > 0).all()
        assert (s0 < 1).all()

    def test_rejects_bad_shares(self):
        df = make_synthetic_data()
        df.loc[df["market_id"] == 1, "share"] = 0.5
        with pytest.raises(ValueError, match="shares summing above 1"):
            MergerData(df, firm_col="firm", market_col="market_id",
                       price_col="price", share_col="share")


class TestOwnershipMatrix:
    def test_pre_merger(self):
        firms = np.array(["A", "B", "C"])
        omega = ownership_matrix(firms)
        np.testing.assert_array_equal(omega, np.eye(3))

    def test_post_merger(self):
        firms = np.array(["A", "B", "C"])
        omega = ownership_matrix(firms, merging=("A", "B"))
        expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_array_equal(omega, expected)


class TestLogitShares:
    def test_shares_sum_to_less_than_one(self):
        model = Logit()
        model.alpha = 2.0
        delta_base = np.array([1.5, 1.4, 0.8])
        prices = np.array([2.0, 1.9, 2.1])
        firms = np.array(["A", "B", "C"])
        shares = model.shares_from_delta_base(delta_base, prices, firms)
        assert shares.sum() < 1.0
        assert (shares > 0).all()

    def test_jacobian_diagonal_negative(self):
        model = Logit()
        model.alpha = 2.0
        shares = np.array([0.1, 0.1, 0.05])
        firms = np.array(["A", "B", "C"])
        jac = model.jacobian_from_shares(shares, firms)
        assert (np.diag(jac) < 0).all()
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert jac[i, j] > 0


class TestNestedLogitShares:
    def test_shares_sum_to_less_than_one(self):
        model = NestedLogit(nests={"g1": ["A", "B"], "g2": ["C"]})
        model.alpha = 2.0
        model.sigma = 0.7
        delta_base = np.array([1.5, 1.4, 0.8])
        prices = np.array([2.0, 1.9, 2.1])
        firms = np.array(["A", "B", "C"])
        shares = model.shares_from_delta_base(delta_base, prices, firms)
        assert shares.sum() < 1.0
        assert (shares > 0).all()

    def test_within_nest_substitution_stronger(self):
        """Within-nest cross-derivatives should be larger than cross-nest."""
        model = NestedLogit(nests={"g1": ["A", "B"], "g2": ["C"]})
        model.alpha = 2.0
        model.sigma = 0.7
        shares = np.array([0.1, 0.1, 0.05])
        firms = np.array(["A", "B", "C"])
        jac = model.jacobian_from_shares(shares, firms)
        within_nest = jac[0, 1]   # A -> B (same nest)
        cross_nest = jac[0, 2]    # A -> C (different nest)
        assert within_nest > cross_nest

    def test_sigma_zero_recovers_logit(self):
        """With sigma=0, nested logit should match plain logit."""
        nlogit = NestedLogit(nests={"g1": ["A", "B"], "g2": ["C"]})
        nlogit.alpha = 2.0
        nlogit.sigma = 0.0
        logit = Logit()
        logit.alpha = 2.0

        delta_base = np.array([1.5, 1.4, 0.8])
        prices = np.array([2.0, 1.9, 2.1])
        firms = np.array(["A", "B", "C"])

        s_nl = nlogit.shares_from_delta_base(delta_base, prices, firms)
        s_l = logit.shares_from_delta_base(delta_base, prices, firms)
        np.testing.assert_allclose(s_nl, s_l, atol=1e-10)
