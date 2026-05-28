from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from merger_simulator.data import MergerData


@dataclass
class EstimationResult:
    """Stores demand estimation output."""

    params: dict[str, float]
    std_errors: dict[str, float]
    t_stats: dict[str, float]
    residuals: np.ndarray
    first_stage_f: Optional[float] = None
    r_squared: Optional[float] = None
    n_obs: int = 0
    model_object: Any = None

    def summary(self) -> pd.DataFrame:
        rows = []
        for k in self.params:
            rows.append({
                "parameter": k,
                "estimate": self.params[k],
                "std_error": self.std_errors.get(k, np.nan),
                "t_stat": self.t_stats.get(k, np.nan),
            })
        df = pd.DataFrame(rows).set_index("parameter")
        return df


class DemandModel(ABC):
    """Abstract base class for demand models used in merger simulation."""

    @abstractmethod
    def estimate(self, data: MergerData, instruments: pd.DataFrame,
                 endog_cols: list[str], exog_cols: list[str]) -> EstimationResult:
        ...

    @abstractmethod
    def predict_shares(self, data: MergerData, prices: np.ndarray,
                       market_id: Any) -> np.ndarray:
        ...

    @abstractmethod
    def jacobian(self, data: MergerData, market_id: Any) -> np.ndarray:
        """Return the J x J matrix of share derivatives: ds_j/dp_k."""
        ...

    def elasticity_matrix(self, data: MergerData, market_id: Any) -> np.ndarray:
        """Own- and cross-price elasticities: e_{jk} = (ds_j/dp_k) * (p_k/s_j)."""
        mkt = data.market_data(market_id)
        jac = self.jacobian(data, market_id)
        prices = mkt[data.price_col].values
        shares = mkt[data.share_col].values
        elas = jac * prices[np.newaxis, :] / shares[:, np.newaxis]
        return elas

    def diversion_ratios(self, data: MergerData, market_id: Any) -> np.ndarray:
        """Diversion ratio D_{jk} = -(ds_k/dp_j) / (ds_j/dp_j)."""
        jac = self.jacobian(data, market_id)
        diag = np.diag(jac)
        div = -jac / diag[:, np.newaxis]
        np.fill_diagonal(div, 0.0)
        return div
