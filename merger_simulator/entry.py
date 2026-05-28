from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.optimize import fsolve

from merger_simulator.data import MergerData
from merger_simulator.demand.base import DemandModel


@dataclass
class EntryResult:
    """Results from entry analysis for a single market."""

    market_id: Any
    variable_profit: float
    entry_probability_pre: float
    entry_probability_post: float
    entered: bool
    equilibrium_prices: np.ndarray | None = None


@dataclass
class EntryAnalysis:
    """Aggregate entry analysis results across markets."""

    results: list[EntryResult]

    @property
    def n_markets(self) -> int:
        return len(self.results)

    @property
    def n_entered(self) -> int:
        return sum(1 for r in self.results if r.entered)

    @property
    def entry_rate(self) -> float:
        return self.n_entered / self.n_markets if self.n_markets > 0 else 0.0

    @property
    def mean_pre_prob(self) -> float:
        return float(np.mean([r.entry_probability_pre for r in self.results]))

    @property
    def mean_post_prob(self) -> float:
        return float(np.mean([r.entry_probability_post for r in self.results]))

    def summary(self) -> pd.DataFrame:
        data = [{
            "market_id": r.market_id,
            "variable_profit": r.variable_profit,
            "entry_prob_pre": r.entry_probability_pre,
            "entry_prob_post": r.entry_probability_post,
            "entered": r.entered,
        } for r in self.results]
        return pd.DataFrame(data)


def entry_probability(
    variable_profit: float | np.ndarray,
    mu: float,
    sigma: float,
) -> float | np.ndarray:
    """P(entry) = P(F <= profit) where log(F) ~ N(mu, sigma^2).

    P(F <= pi) = Phi((log(pi) - mu) / sigma)
    """
    profit = np.asarray(variable_profit)
    prob = np.where(
        profit > 0,
        norm.cdf((np.log(np.maximum(profit, 1e-30)) - mu) / sigma),
        0.0,
    )
    return float(prob) if np.ndim(variable_profit) == 0 else prob


def compute_entrant_profit(
    model: DemandModel,
    data: MergerData,
    market_id: Any,
    entrant_delta_base: float,
    entrant_firm: str,
    entrant_mc_base: float,
    entrant_cost_residual: float = 0.0,
    gamma2: float = 0.0,
    merger: bool = False,
    merging_firms: tuple[str, str] | None = None,
    merging_gamma2: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute variable profit for a potential entrant in a market.

    Solves the Nash equilibrium with the entrant added, then returns
    (entrant_profit, equilibrium_prices, equilibrium_shares).
    """
    mkt = data.market_data(market_id)
    incumbent_delta_base = mkt[model._delta_base_col].values
    incumbent_firms = mkt[data.firm_col].values
    incumbent_mc_base = mkt["_mc_base"].values if "_mc_base" in mkt.columns else mkt["_mc"].values
    incumbent_cost_res = mkt["_cost_residual"].values if "_cost_residual" in mkt.columns else np.zeros(len(mkt))
    mkt_size = mkt[data.market_size_col].values[0] if data.market_size_col else 1.0

    all_delta_base = np.append(incumbent_delta_base, entrant_delta_base)
    all_firms = np.append(incumbent_firms, entrant_firm)
    all_mc_base = np.append(incumbent_mc_base, entrant_mc_base)
    all_cost_res = np.append(incumbent_cost_res, entrant_cost_residual)

    from merger_simulator.merger import ownership_matrix as _om

    if merger and merging_firms:
        omega = _om(all_firms, merging=merging_firms)
    else:
        omega = _om(all_firms)

    alpha = model.alpha
    sigma = model.sigma if hasattr(model, 'sigma') else 0.0
    eff_gamma2 = merging_gamma2 if merging_gamma2 is not None else gamma2

    def foc(prices):
        shares = model.shares_from_delta_base(all_delta_base, prices, all_firms)
        quantities = shares * mkt_size

        mc = np.copy(all_mc_base)
        for i in range(len(all_firms)):
            g2 = eff_gamma2 if (merger and merging_firms and all_firms[i] in merging_firms) else gamma2
            mc[i] = all_mc_base[i] + g2 * quantities[i] + all_cost_res[i]

        jac = model.jacobian_from_shares(shares, all_firms)
        markup = -np.linalg.solve(omega * jac, shares)
        return prices - mc - markup

    p0 = all_mc_base + all_cost_res + 1.0 / (alpha * 0.95)
    eq_prices = fsolve(foc, p0)
    eq_shares = model.shares_from_delta_base(all_delta_base, eq_prices, all_firms)
    eq_quantities = eq_shares * mkt_size

    entrant_idx = len(all_firms) - 1
    g2_ent = gamma2
    eq_mc_ent = all_mc_base[entrant_idx] + g2_ent * eq_quantities[entrant_idx] + all_cost_res[entrant_idx]
    entrant_profit = (eq_prices[entrant_idx] - eq_mc_ent) * eq_quantities[entrant_idx]

    return entrant_profit, eq_prices, eq_shares


def simulate_entry(
    model: DemandModel,
    data: MergerData,
    non_entry_markets: list[Any],
    entrant_firm: str,
    entrant_delta_base: float,
    entrant_mc_base_fn: callable,
    fixed_cost_mu: float,
    fixed_cost_sigma: float,
    gamma2: float = 0.0,
    merger: bool = False,
    merging_firms: tuple[str, str] | None = None,
    merging_gamma2: float | None = None,
    seed: int = 42,
) -> EntryAnalysis:
    """Run entry analysis across markets where the entrant is not present.

    Parameters
    ----------
    entrant_mc_base_fn : callable(market_id) -> float
        Returns the entrant's mc_base for a given market.
    fixed_cost_mu, fixed_cost_sigma : log-normal parameters for fixed costs.
    """
    rng = np.random.default_rng(seed)
    results = []

    for mid in non_entry_markets:
        mc_base = entrant_mc_base_fn(mid)

        profit_pre, _, _ = compute_entrant_profit(
            model, data, mid, entrant_delta_base, entrant_firm,
            mc_base, gamma2=gamma2, merger=False,
        )

        profit_post, eq_prices, _ = compute_entrant_profit(
            model, data, mid, entrant_delta_base, entrant_firm,
            mc_base, gamma2=gamma2, merger=merger,
            merging_firms=merging_firms, merging_gamma2=merging_gamma2,
        )

        prob_pre = entry_probability(profit_pre, fixed_cost_mu, fixed_cost_sigma)
        prob_post = entry_probability(profit_post, fixed_cost_mu, fixed_cost_sigma)

        u = rng.uniform()
        entered = u < prob_post

        results.append(EntryResult(
            market_id=mid,
            variable_profit=profit_post,
            entry_probability_pre=prob_pre,
            entry_probability_post=prob_post,
            entered=entered,
            equilibrium_prices=eq_prices if entered else None,
        ))

    return EntryAnalysis(results=results)
