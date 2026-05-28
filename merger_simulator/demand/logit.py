from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from merger_simulator.data import MergerData
from merger_simulator.demand.base import DemandModel, EstimationResult
from merger_simulator.estimation import estimate_iv2sls


class Logit(DemandModel):
    """Plain logit demand model (Berry 1994 inversion).

    Dependent variable: delta_j = ln(s_j) - ln(s_0)
    Estimated equation: delta_j = x_j * beta - alpha * p_j + xi_j
    """

    def __init__(self) -> None:
        self.alpha: float = 0.0
        self.result: EstimationResult | None = None
        self._delta_col = "_logit_delta"
        self._delta_base_col = "_logit_delta_base"

    def estimate(
        self,
        data: MergerData,
        instruments: pd.DataFrame,
        endog_cols: list[str] | None = None,
        exog_cols: list[str] | None = None,
    ) -> EstimationResult:
        if data.share_col is None:
            raise ValueError("Logit requires share data.")

        df = data.df
        outside = data.outside_shares()
        df = df.merge(outside.rename("_s0"), left_on=data.market_col, right_index=True, how="left")
        delta = np.log(df[data.share_col]) - np.log(df["_s0"])

        if endog_cols is None:
            endog_cols = [data.price_col]
        if exog_cols is None:
            exog_cols = ["const"] + data.product_chars
            if "const" not in df.columns:
                df["const"] = 1

        dep = delta
        exog = df[exog_cols]
        endog = df[endog_cols]
        iv = instruments if isinstance(instruments, pd.DataFrame) else instruments.df

        self.result = estimate_iv2sls(dep, exog, endog, iv)
        self.alpha = -self.result.params[data.price_col]

        data.df[self._delta_col] = dep.values if hasattr(dep, 'values') else dep
        data.df[self._delta_base_col] = data.df[self._delta_col] + self.alpha * data.df[data.price_col]

        if "_s0" in data.df.columns:
            data.df.drop(columns=["_s0"], inplace=True, errors="ignore")
        df.drop(columns=["_s0"], inplace=True, errors="ignore")

        return self.result

    def predict_shares(
        self, data: MergerData, prices: np.ndarray, market_id: Any
    ) -> np.ndarray:
        mkt = data.market_data(market_id)
        delta_base = mkt[self._delta_base_col].values
        delta = delta_base - self.alpha * prices
        exp_delta = np.exp(delta)
        denom = 1.0 + exp_delta.sum()
        return exp_delta / denom

    def jacobian(self, data: MergerData, market_id: Any) -> np.ndarray:
        """ds_j/dp_k for logit: diagonal = -alpha*s_j*(1-s_j), off-diag = alpha*s_j*s_k."""
        mkt = data.market_data(market_id)
        s = mkt[data.share_col].values
        n = len(s)
        jac = np.outer(s, s) * self.alpha
        np.fill_diagonal(jac, -self.alpha * s * (1 - s))
        return jac

    def shares_from_delta_base(
        self, delta_base: np.ndarray, prices: np.ndarray, firms: np.ndarray
    ) -> np.ndarray:
        delta = delta_base - self.alpha * prices
        exp_delta = np.exp(delta)
        return exp_delta / (1.0 + exp_delta.sum())

    def jacobian_from_shares(
        self, shares: np.ndarray, firms: np.ndarray
    ) -> np.ndarray:
        jac = np.outer(shares, shares) * self.alpha
        np.fill_diagonal(jac, -self.alpha * shares * (1 - shares))
        return jac

    def compute_delta_base(self, data: MergerData) -> None:
        """Recompute delta and delta_base from current shares (useful after re-estimation)."""
        outside = data.outside_shares()
        s0 = data.df[data.market_col].map(outside)
        data.df[self._delta_col] = np.log(data.df[data.share_col]) - np.log(s0)
        data.df[self._delta_base_col] = data.df[self._delta_col] + self.alpha * data.df[data.price_col]
