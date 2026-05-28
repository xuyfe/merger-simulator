from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from merger_simulator.data import MergerData


class Instruments:
    """Container for instrument columns that can be combined with +."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def __add__(self, other: Instruments) -> Instruments:
        combined = pd.concat([self.df, other.df], axis=1)
        return Instruments(combined)

    def __repr__(self) -> str:
        return f"Instruments({self.columns})"


def hausman(
    data: MergerData,
    time_col: str,
    cross_section_col: str | None = None,
    col_name: str = "iv_hausman",
) -> Instruments:
    """Hausman instrument: mean price of same firm in other markets within the same time period."""
    df = data.df
    firm = data.firm_col
    price = data.price_col

    group_sum = df.groupby([firm, time_col])[price].transform("sum")
    group_n = df.groupby([firm, time_col])[price].transform("count")
    iv = (group_sum - df[price]) / (group_n - 1)
    return Instruments(pd.DataFrame({col_name: iv}, index=df.index))


def rival_cost_shifters(
    data: MergerData,
    col_name: str = "rival_cost_shifter",
) -> Instruments:
    """Average cost shifter of rival firms in the same market."""
    df = data.df
    mkt = data.market_col
    cols = {}

    for cs_col in data.cost_shifter_cols:
        total = df.groupby(mkt)[cs_col].transform("sum")
        n = df.groupby(mkt)[cs_col].transform("count")
        name = f"{col_name}_{cs_col}" if len(data.cost_shifter_cols) > 1 else col_name
        cols[name] = (total - df[cs_col]) / (n - 1)

    return Instruments(pd.DataFrame(cols, index=df.index))


def blp_instruments(
    data: MergerData,
    include_num_firms: bool = True,
    include_rival_chars: bool = True,
) -> Instruments:
    """BLP-style instruments: number of firms, sum/mean of rival characteristics."""
    df = data.df
    mkt = data.market_col
    cols = {}

    if include_num_firms:
        cols["num_firms"] = df.groupby(mkt)[data.firm_col].transform("count")

    if include_rival_chars and data.product_chars:
        for char in data.product_chars:
            total = df.groupby(mkt)[char].transform("sum")
            n = df.groupby(mkt)[char].transform("count")
            cols[f"rival_{char}"] = (total - df[char]) / (n - 1)

    return Instruments(pd.DataFrame(cols, index=df.index))


def within_nest_rival_cost(
    data: MergerData,
    model: Any = None,
    col_name: str = "within_nest_rival_cost",
) -> Instruments:
    """Within-nest rival cost shifter (for nested logit).

    If data.nest_col is not set, pass the NestedLogit model to auto-populate it.
    """
    if data.nest_col is None and model is not None and hasattr(model, "apply_nests"):
        model.apply_nests(data)
    if data.nest_col is None:
        raise ValueError(
            "nest_col must be set on MergerData, or pass a NestedLogit model."
        )
    df = data.df
    mkt = data.market_col
    nest = data.nest_col
    cols = {}

    for cs_col in data.cost_shifter_cols:
        nest_sum = df.groupby([mkt, nest])[cs_col].transform("sum")
        name = f"{col_name}_{cs_col}" if len(data.cost_shifter_cols) > 1 else col_name
        cols[name] = nest_sum - df[cs_col]

    return Instruments(pd.DataFrame(cols, index=df.index))
