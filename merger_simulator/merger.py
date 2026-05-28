from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from merger_simulator.data import MergerData
from merger_simulator.demand.base import DemandModel


@dataclass
class EquilibriumResult:
    """Output from an equilibrium solve for a single market."""

    prices: np.ndarray
    shares: np.ndarray
    markups: np.ndarray
    profits: np.ndarray
    firms: np.ndarray
    converged: bool = True
    iterations: int = 0


@dataclass
class MergerResult:
    """Full merger simulation output across all markets."""

    pre: pd.DataFrame
    post: pd.DataFrame

    @property
    def price_changes(self) -> pd.DataFrame:
        df = self.post[["firm", "market_id", "price"]].copy()
        df = df.rename(columns={"price": "post_price"})
        df["pre_price"] = self.pre["price"].values
        df["change"] = df["post_price"] - df["pre_price"]
        df["pct_change"] = df["change"] / df["pre_price"] * 100
        return df

    def summary(self) -> pd.DataFrame:
        pc = self.price_changes
        return pc.groupby("firm").agg(
            mean_pct_change=("pct_change", "mean"),
            median_pct_change=("pct_change", "median"),
            min_pct_change=("pct_change", "min"),
            max_pct_change=("pct_change", "max"),
        ).round(4)


def ownership_matrix(
    firms: np.ndarray,
    merging: tuple[str, str] | None = None,
    ownership: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Build an ownership matrix.

    Parameters
    ----------
    merging : pair of firm names that merge (shorthand for simple 2-firm mergers)
    ownership : general ownership mapping (owner -> [firms])
    """
    n = len(firms)

    if ownership is not None:
        firm_to_owner = {}
        for owner, members in ownership.items():
            for f in members:
                firm_to_owner[f] = owner
        for f in firms:
            if f not in firm_to_owner:
                firm_to_owner[f] = f
    elif merging is not None:
        firm_to_owner = {}
        merged_label = f"{merging[0]}+{merging[1]}"
        for f in firms:
            if f in merging:
                firm_to_owner[f] = merged_label
            else:
                firm_to_owner[f] = f
    else:
        firm_to_owner = {f: f for f in firms}

    omega = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            omega[i, j] = 1.0 if firm_to_owner[firms[i]] == firm_to_owner[firms[j]] else 0.0
    return omega


def solve_equilibrium(
    model: DemandModel,
    data: MergerData,
    market_id: Any,
    mc: np.ndarray,
    merging: tuple[str, str] | None = None,
    ownership_map: dict[str, list[str]] | None = None,
    method: str = "fsolve",
    tol: float = 1e-8,
    max_iter: int = 10000,
    damping: float = 0.1,
) -> EquilibriumResult:
    """Solve for Bertrand-Nash equilibrium prices in a single market."""
    mkt = data.market_data(market_id)
    firms = mkt[data.firm_col].values
    n = len(firms)
    omega = ownership_matrix(firms, merging=merging, ownership=ownership_map)
    delta_base = mkt[model._delta_base_col].values

    if method == "fsolve":
        return _solve_fsolve(model, data, delta_base, firms, mc, omega, n, tol)
    elif method == "fixedpoint":
        return _solve_fixedpoint(model, data, delta_base, firms, mc, omega, n, tol, max_iter, damping)
    else:
        raise ValueError(f"Unknown method: {method}")


def _solve_fsolve(model, data, delta_base, firms, mc, omega, n, tol):
    def foc_residual(prices):
        shares = model.shares_from_delta_base(delta_base, prices, firms)
        jac = model.jacobian_from_shares(shares, firms) if hasattr(model, 'jacobian_from_shares') else model._jacobian_from_shares(shares, firms)
        markup = -np.linalg.solve(omega * jac, shares)
        return prices - mc - markup

    p0 = mc + 0.5
    result = fsolve(foc_residual, p0, full_output=True)
    prices = result[0]
    info = result[1]

    shares = model.shares_from_delta_base(delta_base, prices, firms) if hasattr(model, 'shares_from_delta_base') else model._shares_from_delta_base(delta_base, prices, firms)
    jac = model.jacobian_from_shares(shares, firms) if hasattr(model, 'jacobian_from_shares') else model._jacobian_from_shares(shares, firms)
    markups = -np.linalg.solve(omega * jac, shares)
    profits = markups * shares

    return EquilibriumResult(
        prices=prices, shares=shares, markups=markups,
        profits=profits, firms=firms, converged=True,
    )


def _solve_fixedpoint(model, data, delta_base, firms, mc, omega, n, tol, max_iter, damping):
    prices = mc + 0.5
    for it in range(max_iter):
        shares = model.shares_from_delta_base(delta_base, prices, firms) if hasattr(model, 'shares_from_delta_base') else model._shares_from_delta_base(delta_base, prices, firms)
        jac = model.jacobian_from_shares(shares, firms) if hasattr(model, 'jacobian_from_shares') else model._jacobian_from_shares(shares, firms)
        markups = -np.linalg.solve(omega * jac, shares)
        new_prices = mc + markups
        diff = np.max(np.abs(new_prices - prices))
        prices = damping * new_prices + (1 - damping) * prices
        if diff < tol:
            break

    shares = model.shares_from_delta_base(delta_base, prices, firms) if hasattr(model, 'shares_from_delta_base') else model._shares_from_delta_base(delta_base, prices, firms)
    jac = model.jacobian_from_shares(shares, firms) if hasattr(model, 'jacobian_from_shares') else model._jacobian_from_shares(shares, firms)
    markups = -np.linalg.solve(omega * jac, shares)
    profits = markups * shares

    return EquilibriumResult(
        prices=prices, shares=shares, markups=markups,
        profits=profits, firms=firms, converged=(diff < tol), iterations=it + 1,
    )


def simulate_merger(
    model: DemandModel,
    data: MergerData,
    merging_firms: tuple[str, str],
    method: str = "fsolve",
    tol: float = 1e-8,
    max_iter: int = 10000,
    damping: float = 0.1,
    mc_col: str = "_mc",
) -> MergerResult:
    """Run a full merger simulation across all markets.

    Requires that marginal costs have already been recovered (see costs.recover_marginal_costs).
    """
    if mc_col not in data.df.columns:
        raise ValueError(f"Column '{mc_col}' not found. Run recover_marginal_costs first.")

    pre_rows = []
    post_rows = []

    for mid in data.markets:
        mkt = data.market_data(mid)
        firms = mkt[data.firm_col].values
        mc = mkt[mc_col].values
        prices_pre = mkt[data.price_col].values
        shares_pre = mkt[data.share_col].values
        markups_pre = mkt["_markup"].values if "_markup" in mkt.columns else prices_pre - mc

        eq = solve_equilibrium(
            model, data, mid, mc,
            merging=merging_firms,
            method=method, tol=tol, max_iter=max_iter, damping=damping,
        )

        for i in range(len(firms)):
            pre_rows.append({
                "market_id": mid, "firm": firms[i],
                "price": prices_pre[i], "share": shares_pre[i],
                "markup": markups_pre[i], "mc": mc[i],
            })
            mkt_size = mkt[data.market_size_col].values[0] if data.market_size_col else 1.0
            post_rows.append({
                "market_id": mid, "firm": firms[i],
                "price": eq.prices[i], "share": eq.shares[i],
                "markup": eq.markups[i], "mc": mc[i],
                "profit": eq.markups[i] * eq.shares[i] * mkt_size,
            })

    pre_df = pd.DataFrame(pre_rows)
    post_df = pd.DataFrame(post_rows)
    return MergerResult(pre=pre_df, post=post_df)
