from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

from merger_simulator.data import MergerData
from merger_simulator.demand.base import DemandModel


def recover_marginal_costs(
    model: DemandModel,
    data: MergerData,
    ownership: Optional[dict[str, list[str]]] = None,
) -> np.ndarray:
    """Recover marginal costs from Bertrand-Nash FOC: p = mc + markup.

    markup = -[Omega .* Jacobian]^{-1} @ s

    Parameters
    ----------
    ownership : dict mapping owner name -> list of firm names they own.
        If None, each firm owns only itself (pre-merger).
    """
    all_mc = np.full(len(data.df), np.nan)
    all_markup = np.full(len(data.df), np.nan)

    for mid in data.markets:
        mkt = data.market_data(mid)
        idx = mkt.index
        prices = mkt[data.price_col].values
        shares = mkt[data.share_col].values
        firms = mkt[data.firm_col].values

        omega = _ownership_matrix(firms, ownership)
        jac = model.jacobian(data, mid)

        markup = -np.linalg.solve(omega * jac, shares)
        mc = prices - markup

        for i, ix in enumerate(idx):
            pos = data.df.index.get_loc(ix)
            all_mc[pos] = mc[i]
            all_markup[pos] = markup[i]

    data.add_column("_mc", all_mc)
    data.add_column("_markup", all_markup)
    return all_mc


def _ownership_matrix(
    firms: np.ndarray,
    ownership: Optional[dict[str, list[str]]] = None,
) -> np.ndarray:
    n = len(firms)
    if ownership is None:
        omega = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                omega[i, j] = 1.0 if firms[i] == firms[j] else 0.0
        return omega

    firm_to_owner = {}
    for owner, members in ownership.items():
        for f in members:
            firm_to_owner[f] = owner
    for f in firms:
        if f not in firm_to_owner:
            firm_to_owner[f] = f

    omega = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            omega[i, j] = 1.0 if firm_to_owner[firms[i]] == firm_to_owner[firms[j]] else 0.0
    return omega


class CostFunction:
    """Parametric cost model: mc = gamma_0 + gamma_1*Z + gamma_2*Q + eta.

    Estimated via IV-2SLS on the recovered marginal costs.
    """

    def __init__(self) -> None:
        self.params: dict[str, float] = {}
        self.residuals: np.ndarray = np.array([])
        self.model_object: Any = None

    def estimate(
        self,
        data: MergerData,
        quantity_col: str | None = None,
        instruments: pd.DataFrame | None = None,
        exog_cols: list[str] | None = None,
    ) -> None:
        df = data.df
        if "_mc" not in df.columns:
            raise ValueError("Run recover_marginal_costs first.")

        if exog_cols is None:
            exog_cols = ["const"] + data.cost_shifter_cols
        if "const" not in df.columns:
            df["const"] = 1

        dep = df["_mc"]
        exog = df[exog_cols]

        if quantity_col and instruments is not None:
            endog = df[[quantity_col]]
            iv = instruments if isinstance(instruments, pd.DataFrame) else instruments.df
            model = IV2SLS(dep, exog, endog, iv).fit(cov_type="robust")
        else:
            from linearmodels.iv import IV2SLS as _IV
            endog = None
            model = _IV(dep, exog, None, None).fit(cov_type="robust")

        self.params = dict(model.params)
        self.residuals = model.resids.values
        self.model_object = model

        data.add_column("_mc_base", model.fitted_values.values - (
            self.params.get(quantity_col, 0) * df[quantity_col].values if quantity_col else 0
        ))
        data.add_column("_cost_residual", self.residuals)

    def predict_mc(
        self,
        cost_shifters: dict[str, float],
        quantity: float,
        residual: float = 0.0,
        quantity_col: str = "quantity",
        efficiency_scale: float = 1.0,
    ) -> float:
        mc = self.params.get("const", 0.0) + residual
        for col, val in cost_shifters.items():
            mc += self.params.get(col, 0.0) * val
        mc += self.params.get(quantity_col, 0.0) * efficiency_scale * quantity
        return mc
