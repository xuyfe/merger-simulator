from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from merger_simulator.data import MergerData
from merger_simulator.demand.base import DemandModel, EstimationResult
from merger_simulator.estimation import estimate_iv2sls


class NestedLogit(DemandModel):
    """Nested logit demand model.

    Berry inversion:
        ln(s_j) - ln(s_0) = x_j*beta - alpha*p_j + sigma*ln(s_{j|g}) + xi_j

    Parameters
    ----------
    nests : dict mapping nest name -> list of firm names in that nest.
        Firms not listed are each placed in their own singleton nest.
    """

    def __init__(self, nests: dict[str, list[str]]) -> None:
        self.nests = nests
        self.alpha: float = 0.0
        self.sigma: float = 0.0
        self.result: EstimationResult | None = None
        self._delta_col = "_nlogit_delta"
        self._delta_base_col = "_nlogit_delta_base"

    def _firm_to_nest(self, firms: list[str] | np.ndarray) -> dict[str, str]:
        mapping = {}
        for nest_name, members in self.nests.items():
            for f in members:
                mapping[f] = nest_name
        for f in firms:
            if f not in mapping:
                mapping[f] = f"_singleton_{f}"
        return mapping

    def _compute_within_nest_share(self, data: MergerData) -> pd.Series:
        df = data.df
        nest_map = self._firm_to_nest(data.firms)
        nest_labels = df[data.firm_col].map(nest_map)
        nest_total = df.groupby([data.market_col, nest_labels])[data.share_col].transform("sum")
        return df[data.share_col] / nest_total

    def apply_nests(self, data: MergerData) -> None:
        """Populate the nest column on data from this model's nest structure."""
        nest_map = self._firm_to_nest(data.firms)
        data.df["_nest"] = data.df[data.firm_col].map(nest_map)
        data.nest_col = "_nest"

    def estimate(
        self,
        data: MergerData,
        instruments: pd.DataFrame,
        endog_cols: list[str] | None = None,
        exog_cols: list[str] | None = None,
    ) -> EstimationResult:
        if data.share_col is None:
            raise ValueError("Nested logit requires share data.")

        self.apply_nests(data)
        df = data.df
        outside = data.outside_shares()
        s0 = df[data.market_col].map(outside)
        delta = np.log(df[data.share_col].values) - np.log(s0.values)

        within_share = self._compute_within_nest_share(data)
        ln_within = np.log(within_share.values)
        df["_ln_within_nest_share"] = ln_within

        if endog_cols is None:
            endog_cols = [data.price_col, "_ln_within_nest_share"]
        if exog_cols is None:
            exog_cols = ["const"] + data.product_chars
            if "const" not in df.columns:
                df["const"] = 1

        dep = pd.Series(delta, index=df.index)
        exog = df[exog_cols]
        endog = df[endog_cols]
        iv = instruments if isinstance(instruments, pd.DataFrame) else instruments.df

        self.result = estimate_iv2sls(dep, exog, endog, iv)

        self.alpha = -self.result.params[data.price_col]
        self.sigma = self.result.params["_ln_within_nest_share"]

        data.df[self._delta_col] = delta - self.sigma * ln_within
        data.df[self._delta_base_col] = data.df[self._delta_col] + self.alpha * data.df[data.price_col]

        return self.result

    def predict_shares(
        self, data: MergerData, prices: np.ndarray, market_id: Any
    ) -> np.ndarray:
        mkt = data.market_data(market_id)
        delta_base = mkt[self._delta_base_col].values
        firms = mkt[data.firm_col].values
        return self._shares_from_delta_base(delta_base, prices, firms)

    def _shares_from_delta_base(
        self, delta_base: np.ndarray, prices: np.ndarray, firms: np.ndarray
    ) -> np.ndarray:
        delta = delta_base - self.alpha * prices
        nest_map = self._firm_to_nest(firms)
        sigma = self.sigma

        nests: dict[str, list[int]] = {}
        for i, f in enumerate(firms):
            n = nest_map[f]
            nests.setdefault(n, []).append(i)

        inclusive_values = {}
        for nest_name, indices in nests.items():
            inclusive_values[nest_name] = sum(
                np.exp(delta[i] / (1 - sigma)) for i in indices
            )

        denom = 1.0 + sum(D ** (1 - sigma) for D in inclusive_values.values())

        shares = np.zeros(len(delta))
        for nest_name, indices in nests.items():
            D_g = inclusive_values[nest_name]
            s_g = D_g ** (1 - sigma) / denom
            for i in indices:
                s_jg = np.exp(delta[i] / (1 - sigma)) / D_g
                shares[i] = s_jg * s_g

        return shares

    def jacobian(self, data: MergerData, market_id: Any) -> np.ndarray:
        mkt = data.market_data(market_id)
        s = mkt[data.share_col].values
        firms = mkt[data.firm_col].values
        return self._jacobian_from_shares(s, firms)

    def _jacobian_from_shares(
        self, shares: np.ndarray, firms: np.ndarray
    ) -> np.ndarray:
        n = len(shares)
        alpha = self.alpha
        sigma = self.sigma
        nest_map = self._firm_to_nest(firms)

        nests: dict[str, list[int]] = {}
        for i, f in enumerate(firms):
            nests.setdefault(nest_map[f], []).append(i)

        nest_totals = {}
        for nest_name, indices in nests.items():
            nest_totals[nest_name] = sum(shares[i] for i in indices)

        within_share = np.zeros(n)
        for i, f in enumerate(firms):
            within_share[i] = shares[i] / nest_totals[nest_map[f]]

        jac = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    jac[i, j] = -alpha * shares[i] * (
                        1 / (1 - sigma)
                        - sigma / (1 - sigma) * within_share[i]
                        - shares[i]
                    )
                elif nest_map[firms[i]] == nest_map[firms[j]]:
                    jac[i, j] = alpha * shares[i] * (
                        sigma / (1 - sigma) * within_share[j] + shares[j]
                    )
                else:
                    jac[i, j] = alpha * shares[i] * shares[j]

        return jac

    def jacobian_from_shares(
        self, shares: np.ndarray, firms: np.ndarray
    ) -> np.ndarray:
        """Public version for use in equilibrium solvers with updated shares."""
        return self._jacobian_from_shares(shares, firms)

    def shares_from_delta_base(
        self, delta_base: np.ndarray, prices: np.ndarray, firms: np.ndarray
    ) -> np.ndarray:
        """Public version for use in equilibrium solvers."""
        return self._shares_from_delta_base(delta_base, prices, firms)
