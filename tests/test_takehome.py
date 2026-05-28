"""Integration test: reproduce take-home final results with the package."""
import os
import numpy as np
import pandas as pd
import pytest

from merger_simulator import (
    MergerData, NestedLogit,
    simulate_merger, recover_marginal_costs,
)
from merger_simulator.instruments import (
    hausman, rival_cost_shifters, blp_instruments, within_nest_rival_cost,
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Take_Home_Final", "fast_food.csv"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason="Take-home data not found"
)


@pytest.fixture(scope="module")
def data_and_model():
    df = pd.read_csv(DATA_PATH)

    df["is_tacobell"] = (df["firm"] == "TacoBell").astype(int)
    df["is_sushihut"] = (df["firm"] == "SushiHut").astype(int)
    df["const"] = 1

    data = MergerData(
        df, firm_col="firm", market_col="market_id",
        price_col="price", share_col="share",
        market_size_col="market_size",
        cost_shifter_cols=["cost_shifter"],
        product_chars=["advertising", "is_tacobell", "is_sushihut"],
        firm_id_col="firm_id",
    )

    model = NestedLogit(nests={"legacy": ["McDonalds", "TacoBell"], "new": ["SushiHut"]})
    model.apply_nests(data)

    data_for_blp = MergerData(
        data.df.copy(), firm_col="firm", market_col="market_id",
        price_col="price", share_col="share",
        market_size_col="market_size",
        cost_shifter_cols=["cost_shifter"],
        product_chars=["advertising"],
        firm_id_col="firm_id",
    )

    iv = (
        hausman(data, time_col="quarter_id")
        + rival_cost_shifters(data)
        + blp_instruments(data_for_blp, include_rival_chars=True)
        + within_nest_rival_cost(data)
    )
    result = model.estimate(
        data, instruments=iv,
        endog_cols=["price", "_ln_within_nest_share"],
        exog_cols=["const", "advertising", "is_tacobell", "is_sushihut"],
    )

    return data, model, result


class TestDemandEstimation:
    def test_alpha(self, data_and_model):
        _, model, _ = data_and_model
        assert abs(model.alpha - 2.0076) < 0.01

    def test_sigma(self, data_and_model):
        _, model, _ = data_and_model
        assert abs(model.sigma - 0.7761) < 0.01

    def test_advertising_positive(self, data_and_model):
        _, _, result = data_and_model
        assert result.params["advertising"] > 0.4

    def test_sushihut_fe_negative(self, data_and_model):
        _, _, result = data_and_model
        assert result.params["is_sushihut"] < -0.4


class TestMarginalCosts:
    def test_mean_markup(self, data_and_model):
        data, model, _ = data_and_model
        recover_marginal_costs(model, data)
        mean_markup = data.df["_markup"].mean()
        assert abs(mean_markup - 0.30) < 0.02

    def test_mc_by_firm(self, data_and_model):
        data, model, _ = data_and_model
        mc_by_firm = data.df.groupby("firm")["_mc"].mean()
        assert abs(mc_by_firm["McDonalds"] - 2.12) < 0.05
        assert abs(mc_by_firm["SushiHut"] - 1.81) < 0.05
        assert abs(mc_by_firm["TacoBell"] - 2.16) < 0.05


class TestMergerSimulation:
    def test_price_increases(self, data_and_model):
        data, model, _ = data_and_model
        if "_mc" not in data.df.columns:
            recover_marginal_costs(model, data)

        result = simulate_merger(
            model, data,
            merging_firms=("McDonalds", "TacoBell"),
        )

        summary = result.summary()
        mcd_pct = summary.loc["McDonalds", "mean_pct_change"]
        tb_pct = summary.loc["TacoBell", "mean_pct_change"]
        sh_pct = summary.loc["SushiHut", "mean_pct_change"]

        assert abs(mcd_pct - 12.57) < 1.0, f"McDonalds: {mcd_pct:.2f}%"
        assert abs(tb_pct - 12.95) < 1.0, f"TacoBell: {tb_pct:.2f}%"
        assert sh_pct < 0.5, f"SushiHut should barely change: {sh_pct:.2f}%"
