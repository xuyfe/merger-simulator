from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from merger_simulator.data import MergerData
from merger_simulator.demand.base import DemandModel, EstimationResult
from merger_simulator.estimation import estimate_iv2sls


class LinearDemand(DemandModel):
    """Linear demand system: Q_j = alpha_j + sum_k beta_jk * P_k + epsilon_j.

    Estimated equation-by-equation via IV-2SLS in wide format.
    Each row is a market observation with price/quantity columns for every firm.

    Parameters
    ----------
    firms : list of firm/product identifiers (e.g. ["1","2","3","4","5"])
    price_cols : dict mapping firm -> price column name in the wide DataFrame
    quantity_cols : dict mapping firm -> quantity column name in the wide DataFrame
    """

    def __init__(
        self,
        firms: list[str],
        price_cols: dict[str, str],
        quantity_cols: dict[str, str],
    ) -> None:
        self.firms_list = firms
        self.price_cols = price_cols
        self.quantity_cols = quantity_cols
        self.n_firms = len(firms)
        self.B: np.ndarray | None = None
        self.intercepts: np.ndarray | None = None
        self.results: dict[str, EstimationResult] = {}
        self.residuals: dict[str, np.ndarray] = {}
        self._delta_base_col = "_linear_dummy"

    def estimate(
        self,
        data: MergerData,
        instruments: pd.DataFrame,
        endog_cols: list[str] | None = None,
        exog_cols: list[str] | None = None,
    ) -> EstimationResult:
        """Estimate the system equation-by-equation.

        Parameters
        ----------
        endog_cols : price columns to instrument (all price columns by default)
        exog_cols : exogenous regressors included in every equation (e.g. ["const"])
        """
        df = data.df
        if "const" not in df.columns:
            df["const"] = 1

        if endog_cols is None:
            endog_cols = [self.price_cols[f] for f in self.firms_list]
        if exog_cols is None:
            exog_cols = ["const"]

        iv = instruments if isinstance(instruments, pd.DataFrame) else instruments.df

        self.B = np.zeros((self.n_firms, self.n_firms))
        self.intercepts = np.zeros(self.n_firms)
        all_params: dict[str, float] = {}
        all_se: dict[str, float] = {}
        all_t: dict[str, float] = {}

        for i, firm in enumerate(self.firms_list):
            dep = df[self.quantity_cols[firm]]
            exog = df[exog_cols]
            endog = df[endog_cols]

            result = estimate_iv2sls(dep, exog, endog, iv)
            self.results[firm] = result
            self.residuals[firm] = result.residuals

            self.intercepts[i] = result.params.get("const", 0.0)
            for j, other in enumerate(self.firms_list):
                pcol = self.price_cols[other]
                self.B[i, j] = result.params.get(pcol, 0.0)

            for k, v in result.params.items():
                all_params[f"{firm}_{k}"] = v
                all_se[f"{firm}_{k}"] = result.std_errors.get(k, np.nan)
                all_t[f"{firm}_{k}"] = result.t_stats.get(k, np.nan)

        combined = EstimationResult(
            params=all_params, std_errors=all_se, t_stats=all_t,
            residuals=np.column_stack([self.residuals[f] for f in self.firms_list]),
            n_obs=len(df),
        )
        self.result = combined
        return combined

    def predict_quantities(
        self, df: pd.DataFrame, prices: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        """Predict quantities given current or overridden prices."""
        quantities = {}
        for i, firm in enumerate(self.firms_list):
            q = self.intercepts[i] + self.residuals[firm]
            for j, other in enumerate(self.firms_list):
                p = prices[other] if prices else df[self.price_cols[other]].values
                q = q + self.B[i, j] * p
            quantities[firm] = q
        return quantities

    def predict_shares(self, data: MergerData, prices: np.ndarray, market_id: Any) -> np.ndarray:
        raise NotImplementedError("LinearDemand operates on quantities, not shares.")

    def jacobian(self, data: MergerData, market_id: Any) -> np.ndarray:
        """dQ_j/dP_k = B[j,k] (constant for linear demand)."""
        return self.B.copy()

    def jacobian_from_prices(self, prices: np.ndarray, firms: np.ndarray) -> np.ndarray:
        return self.B.copy()

    def elasticity_matrix(self, data: MergerData, market_id: Any = None,
                          prices: np.ndarray | None = None,
                          quantities: np.ndarray | None = None) -> np.ndarray:
        """e_jk = B[j,k] * P_k / Q_j. For linear demand, elasticities vary by observation."""
        if prices is None or quantities is None:
            raise ValueError("Linear demand elasticities require prices and quantities.")
        elas = self.B * prices[np.newaxis, :] / quantities[:, np.newaxis]
        return elas

    def recover_marginal_costs(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """MC_j = P_j + Q_j / B_jj (from Bertrand-Nash FOC for linear demand)."""
        mc = {}
        for i, firm in enumerate(self.firms_list):
            p = df[self.price_cols[firm]].values
            q = df[self.quantity_cols[firm]].values
            mc[firm] = p + q / self.B[i, i]
        return mc

    def recover_marginal_costs_matrix(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """MC via full matrix inversion: MC = P + B_inv @ Q (multi-product FOC)."""
        B_inv = np.linalg.inv(self.B)
        n = len(df)
        P = np.column_stack([df[self.price_cols[f]].values for f in self.firms_list])
        Q = np.column_stack([df[self.quantity_cols[f]].values for f in self.firms_list])
        MC = P + Q @ B_inv.T
        return {f: MC[:, i] for i, f in enumerate(self.firms_list)}

    def simulate_merger(
        self,
        df: pd.DataFrame,
        merging: tuple[str, str],
        mc: dict[str, np.ndarray] | None = None,
        tol: float = 0.01,
        max_iter: int = 10000,
        damping: float = 0.1,
    ) -> dict:
        """Fixed-point merger simulation for linear demand.

        Merged FOC: p_a = c_a - Q_a/B_aa - (p_b - c_b) * B_ba/B_aa
        Independent FOC: p_j = c_j - Q_j/B_jj
        """
        if mc is None:
            mc = self.recover_marginal_costs(df)

        df_cf = df.copy()
        prices_orig = {f: df[self.price_cols[f]].values.copy() for f in self.firms_list}

        for it in range(max_iter):
            quantities = self.predict_quantities(df_cf)

            p_new = {}
            for firm in self.firms_list:
                i = self.firms_list.index(firm)
                p_j = df_cf[self.price_cols[firm]].values
                c_j = mc[firm]

                if firm in merging:
                    partner = [f for f in merging if f != firm][0]
                    k = self.firms_list.index(partner)
                    p_k = df_cf[self.price_cols[partner]].values
                    c_k = mc[partner]
                    p_new[firm] = c_j - quantities[firm] / self.B[i, i] - \
                        (p_k - c_k) * self.B[k, i] / self.B[i, i]
                else:
                    p_new[firm] = c_j - quantities[firm] / self.B[i, i]

            diff = max(
                np.abs(p_new[f] - df_cf[self.price_cols[f]].values).max()
                for f in self.firms_list
            )

            for f in self.firms_list:
                df_cf[self.price_cols[f]] = (
                    damping * p_new[f] + (1 - damping) * df_cf[self.price_cols[f]].values
                )

            if diff < tol:
                break

        quantities_final = self.predict_quantities(df_cf)

        def profit(firm, d):
            q = self.predict_quantities(d)
            return (d[self.price_cols[firm]].values - mc[firm]) * q[firm]

        return {
            "merger": f"{merging[0]}+{merging[1]}",
            "iterations": it + 1,
            "converged": diff < tol,
            "diff": diff,
            "price_changes": {
                f: {
                    "mean_pct": float(np.mean(
                        (df_cf[self.price_cols[f]].values - prices_orig[f]) / prices_orig[f] * 100
                    )),
                    "prices_post": df_cf[self.price_cols[f]].values,
                }
                for f in self.firms_list
            },
            "profit_changes": {
                f: float(profit(f, df_cf).sum() - profit(f, df).sum())
                for f in self.firms_list
            },
            "df_post": df_cf,
        }
