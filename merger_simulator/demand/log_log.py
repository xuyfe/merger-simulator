from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from merger_simulator.data import MergerData
from merger_simulator.demand.base import DemandModel, EstimationResult
from merger_simulator.estimation import estimate_iv2sls


class LogLogDemand(DemandModel):
    """Log-log (constant elasticity) demand system.

    log(Q_j) = alpha_j + sum_k beta_jk * log(P_k) + controls + epsilon_j

    beta_jk are directly the price elasticities.

    Parameters
    ----------
    firms : list of firm/product identifiers
    price_cols : dict mapping firm -> price column name (linear prices)
    quantity_cols : dict mapping firm -> quantity column name
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
        self.results: dict[str, EstimationResult] = {}
        self.residuals: dict[str, np.ndarray] = {}
        self.models: dict[str, Any] = {}
        self._rhs_cols: list[str] = []
        self._delta_base_col = "_loglog_dummy"

    def estimate(
        self,
        data: MergerData,
        instruments: pd.DataFrame,
        endog_cols: list[str] | None = None,
        exog_cols: list[str] | None = None,
        logprice_cols: dict[str, str] | None = None,
        logquantity_cols: dict[str, str] | None = None,
    ) -> EstimationResult:
        """Estimate log-log demand equation-by-equation.

        Parameters
        ----------
        endog_cols : log-price columns to instrument
        exog_cols : exogenous regressors (e.g. ["const", "avg_pop"])
        logprice_cols : dict firm -> log(price) column name. Created automatically if absent.
        logquantity_cols : dict firm -> log(quantity) column name. Created automatically if absent.
        """
        df = data.df
        if "const" not in df.columns:
            df["const"] = 1

        if logprice_cols is None:
            logprice_cols = {}
            for f in self.firms_list:
                col = f"logp_{f}"
                df[col] = np.log(df[self.price_cols[f]].clip(lower=1e-6))
                logprice_cols[f] = col
        self._logprice_cols = logprice_cols

        if logquantity_cols is None:
            logquantity_cols = {}
            for f in self.firms_list:
                col = f"logq_{f}"
                df[col] = np.log(df[self.quantity_cols[f]].clip(lower=1e-6))
                logquantity_cols[f] = col
        self._logquantity_cols = logquantity_cols

        if endog_cols is None:
            endog_cols = [logprice_cols[f] for f in self.firms_list]
        if exog_cols is None:
            exog_cols = ["const"]
        self._rhs_cols = exog_cols + endog_cols

        iv = instruments if isinstance(instruments, pd.DataFrame) else instruments.df

        self.B = np.zeros((self.n_firms, self.n_firms))
        all_params: dict[str, float] = {}
        all_se: dict[str, float] = {}
        all_t: dict[str, float] = {}

        for i, firm in enumerate(self.firms_list):
            dep = df[logquantity_cols[firm]]
            exog_df = df[exog_cols]
            endog_df = df[endog_cols]

            result = estimate_iv2sls(dep, exog_df, endog_df, iv)
            self.results[firm] = result
            self.residuals[firm] = result.residuals
            self.models[firm] = result.model_object

            for j, other in enumerate(self.firms_list):
                lpcol = logprice_cols[other]
                self.B[i, j] = result.params.get(lpcol, 0.0)

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
        self, df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Predict Q_j = exp(X @ coef + residual) using current prices in df."""
        for f in self.firms_list:
            df[self._logprice_cols[f]] = np.log(df[self.price_cols[f]].clip(lower=1e-6))

        quantities = {}
        for i, firm in enumerate(self.firms_list):
            result = self.results[firm]
            X = df[self._rhs_cols].values
            coefs = np.array([result.params[c] for c in self._rhs_cols])
            logq_hat = X @ coefs + self.residuals[firm]
            quantities[firm] = np.exp(logq_hat)
        return quantities

    def predict_shares(self, data: MergerData, prices: np.ndarray, market_id: Any) -> np.ndarray:
        raise NotImplementedError("LogLogDemand operates on quantities, not shares.")

    def jacobian(self, data: MergerData, market_id: Any) -> np.ndarray:
        """dQ_j/dP_k = B_jk * Q_j / P_k (varies by observation). Not meaningful for single market_id."""
        raise NotImplementedError(
            "Log-log jacobian depends on current prices/quantities. "
            "Use jacobian_at() instead."
        )

    def jacobian_at(
        self, prices: dict[str, np.ndarray], quantities: dict[str, np.ndarray]
    ) -> np.ndarray:
        """dQ_j/dP_k = B_jk * Q_j / P_k for each observation. Returns (n_obs, J, J)."""
        n = len(next(iter(prices.values())))
        J = self.n_firms
        jac = np.zeros((n, J, J))
        for i, fi in enumerate(self.firms_list):
            for j, fj in enumerate(self.firms_list):
                jac[:, i, j] = self.B[i, j] * quantities[fi] / prices[fj]
        return jac

    def recover_marginal_costs(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """MC_j = P_j + P_j / B_jj (from Bertrand-Nash FOC for log-log demand)."""
        mc = {}
        for i, firm in enumerate(self.firms_list):
            p = df[self.price_cols[firm]].values
            mc[firm] = p + p / self.B[i, i]
        return mc

    def simulate_merger(
        self,
        df: pd.DataFrame,
        merging: tuple[str, str],
        mc: dict[str, np.ndarray] | None = None,
        tol: float = 0.01,
        max_iter: int = 10000,
        damping: float = 0.1,
        price_bounds: tuple[float, float] | None = None,
    ) -> dict:
        """Fixed-point merger simulation for log-log demand.

        Merged FOC: p_a = c_a - p_a/B_aa - (p_b - c_b) * (B_ba/B_aa) * (Q_b/Q_a)
        Independent FOC: p_j = c_j - p_j/B_jj
        """
        if mc is None:
            mc = self.recover_marginal_costs(df)

        df_cf = df.copy()
        prices_orig = {f: df[self.price_cols[f]].values.copy() for f in self.firms_list}

        lb = price_bounds[0] if price_bounds else 0.01
        ub = price_bounds[1] if price_bounds else 1e6

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
                    Q_j = quantities[firm]
                    Q_k = quantities[partner]
                    ratio = np.where(Q_j > 1e-20, Q_k / Q_j, 0.0)
                    p_new[firm] = c_j - p_j / self.B[i, i] - \
                        (p_k - c_k) * (self.B[k, i] / self.B[i, i]) * ratio
                else:
                    p_new[firm] = c_j - p_j / self.B[i, i]

                p_new[firm] = np.clip(p_new[firm], lb, ub)

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
