from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from merger_simulator.merger import MergerResult


@dataclass
class WelfareSummary:
    """Welfare metrics comparing pre- and post-merger outcomes."""

    consumer_surplus_change: float
    producer_surplus_change: float
    total_surplus_change: float
    hhi_pre: float
    hhi_post: float
    hhi_change: float
    price_index_change: float
    firm_details: pd.DataFrame

    def __repr__(self) -> str:
        lines = [
            "=== Welfare Summary ===",
            f"Consumer surplus change:  {self.consumer_surplus_change:+.4f}",
            f"Producer surplus change:  {self.producer_surplus_change:+.4f}",
            f"Total surplus change:     {self.total_surplus_change:+.4f}",
            f"HHI pre:  {self.hhi_pre:.0f}",
            f"HHI post: {self.hhi_post:.0f}",
            f"HHI change: {self.hhi_change:+.0f}",
            f"Price index change: {self.price_index_change:+.4f}%",
        ]
        return "\n".join(lines)


def hhi(shares: np.ndarray, firm_labels: np.ndarray,
        ownership: dict[str, str] | None = None) -> float:
    """Compute HHI (0-10000 scale) from market shares.

    Parameters
    ----------
    ownership : optional dict mapping firm -> owner. Shares of firms with
        the same owner are summed before squaring.
    """
    if ownership is not None:
        owner_shares: dict[str, float] = {}
        for s, f in zip(shares, firm_labels):
            owner = ownership.get(f, f)
            owner_shares[owner] = owner_shares.get(owner, 0.0) + s
        total = sum(owner_shares.values())
        return sum((s / total * 100) ** 2 for s in owner_shares.values())
    else:
        total = shares.sum()
        return sum((s / total * 100) ** 2 for s in shares)


def consumer_surplus_logit(
    shares_pre: np.ndarray,
    shares_post: np.ndarray,
    alpha: float,
    market_size: float = 1.0,
) -> float:
    """Exact consumer surplus change for logit demand.

    CS = (M / alpha) * [ln(1 + sum exp(delta_post)) - ln(1 + sum exp(delta_pre))]

    Since we observe shares, we use the log-sum formula:
    CS_change = (M / alpha) * ln[ (1 - sum(s_post)) / (1 - sum(s_pre)) ]
    (sign flipped: higher s_inside means lower outside share means higher CS)

    Actually the exact formula is:
    CS = -(M / alpha) * ln(s_0)  where s_0 is outside share.
    So CS_change = -(M / alpha) * [ln(s_0_post) - ln(s_0_pre)]
                 = (M / alpha) * ln(s_0_pre / s_0_post)
    """
    s0_pre = 1.0 - shares_pre.sum()
    s0_post = 1.0 - shares_post.sum()
    if s0_pre <= 0 or s0_post <= 0:
        return np.nan
    return (market_size / alpha) * np.log(s0_pre / s0_post)


def consumer_surplus_change_approx(
    prices_pre: np.ndarray,
    prices_post: np.ndarray,
    quantities_pre: np.ndarray,
    quantities_post: np.ndarray,
) -> float:
    """First-order approximation of CS change: -sum(q_avg * dp)."""
    dp = prices_post - prices_pre
    q_avg = (quantities_pre + quantities_post) / 2
    return float(-np.sum(q_avg * dp))


def compute_welfare(
    result: MergerResult,
    alpha: float | None = None,
    market_size_col: str | None = None,
    merging_firms: tuple[str, str] | None = None,
) -> WelfareSummary:
    """Compute welfare metrics from a MergerResult.

    Parameters
    ----------
    alpha : price coefficient (needed for exact logit CS). If None, uses approximation.
    merging_firms : which firms merged (for post-merger HHI calculation).
    """
    pre = result.pre
    post = result.post

    # Producer surplus change (sum of profit changes)
    if "profit" in post.columns:
        ps_post = post["profit"].sum()
        ps_pre = (pre["markup"] * pre.get("quantity", pre.get("share", 1))).sum()
        if "quantity" not in pre.columns and "share" in pre.columns:
            mkt_size = 1.0
            ps_pre = (pre["markup"] * pre["share"] * mkt_size).sum()
        ps_change = ps_post - ps_pre
    else:
        ps_change = 0.0

    # Consumer surplus change
    cs_change = 0.0
    markets = pre["market_id"].unique()

    for mid in markets:
        pre_mkt = pre[pre["market_id"] == mid]
        post_mkt = post[post["market_id"] == mid]

        p_pre = pre_mkt["price"].values
        p_post = post_mkt["price"].values
        s_pre = pre_mkt["share"].values if "share" in pre_mkt.columns else None
        s_post = post_mkt["share"].values if "share" in post_mkt.columns else None

        mkt_size = 1.0

        if alpha is not None and s_pre is not None and s_post is not None:
            cs_change += consumer_surplus_logit(s_pre, s_post, alpha, mkt_size)
        elif s_pre is not None:
            cs_change += consumer_surplus_change_approx(p_pre, p_post, s_pre, s_post)

    # HHI
    hhi_values_pre = []
    hhi_values_post = []
    post_ownership = None
    if merging_firms:
        merged_label = f"{merging_firms[0]}+{merging_firms[1]}"
        post_ownership = {merging_firms[0]: merged_label, merging_firms[1]: merged_label}

    for mid in markets:
        pre_mkt = pre[pre["market_id"] == mid]
        post_mkt = post[post["market_id"] == mid]
        s_pre = pre_mkt["share"].values if "share" in pre_mkt.columns else np.ones(len(pre_mkt)) / len(pre_mkt)
        s_post = post_mkt["share"].values if "share" in post_mkt.columns else np.ones(len(post_mkt)) / len(post_mkt)
        firms_pre = pre_mkt["firm"].values
        firms_post = post_mkt["firm"].values

        hhi_values_pre.append(hhi(s_pre, firms_pre))
        hhi_values_post.append(hhi(s_post, firms_post, ownership=post_ownership))

    mean_hhi_pre = float(np.mean(hhi_values_pre))
    mean_hhi_post = float(np.mean(hhi_values_post))

    # Price index change (share-weighted average price change)
    price_idx = 0.0
    weight_total = 0.0
    for mid in markets:
        pre_mkt = pre[pre["market_id"] == mid]
        post_mkt = post[post["market_id"] == mid]
        s = pre_mkt["share"].values if "share" in pre_mkt.columns else np.ones(len(pre_mkt)) / len(pre_mkt)
        dp_pct = (post_mkt["price"].values - pre_mkt["price"].values) / pre_mkt["price"].values * 100
        price_idx += np.sum(s * dp_pct)
        weight_total += s.sum()

    price_idx_change = price_idx / weight_total if weight_total > 0 else 0.0

    # Firm-level details
    firm_details = post.groupby("firm").agg(
        mean_price_post=("price", "mean"),
        mean_markup_post=("markup", "mean"),
        total_profit_post=("profit", "sum") if "profit" in post.columns else ("markup", "sum"),
    ).round(4)
    firm_pre = pre.groupby("firm").agg(
        mean_price_pre=("price", "mean"),
    ).round(4)
    firm_details = firm_details.join(firm_pre)

    return WelfareSummary(
        consumer_surplus_change=cs_change,
        producer_surplus_change=ps_change,
        total_surplus_change=cs_change + ps_change,
        hhi_pre=mean_hhi_pre,
        hhi_post=mean_hhi_post,
        hhi_change=mean_hhi_post - mean_hhi_pre,
        price_index_change=price_idx_change,
        firm_details=firm_details,
    )
