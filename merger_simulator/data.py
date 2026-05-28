from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MergerData:
    """Wraps a DataFrame with column mappings for merger simulation.

    Normalises any dataset to a consistent long-format representation
    (one row per product-market) while keeping the original DataFrame intact.
    """

    df: pd.DataFrame
    firm_col: str
    market_col: str
    price_col: str
    share_col: Optional[str] = None
    quantity_col: Optional[str] = None
    market_size_col: Optional[str] = None
    cost_shifter_cols: list[str] = field(default_factory=list)
    product_chars: list[str] = field(default_factory=list)
    nest_col: Optional[str] = None
    firm_id_col: Optional[str] = None

    def __post_init__(self) -> None:
        if self.share_col is None and self.quantity_col is None:
            raise ValueError("At least one of share_col or quantity_col must be provided.")
        if self.share_col is not None and self.quantity_col is None and self.market_size_col is not None:
            self.df = self.df.copy()
            self.df["_quantity"] = self.df[self.share_col] * self.df[self.market_size_col]
            self.quantity_col = "_quantity"
        if self.firm_id_col is None:
            self.df = self.df.copy()
            firms = self.df[self.firm_col].unique()
            firm_map = {f: i + 1 for i, f in enumerate(firms)}
            self.df["_firm_id"] = self.df[self.firm_col].map(firm_map)
            self.firm_id_col = "_firm_id"
        self._validate()

    def _validate(self) -> None:
        required = [self.firm_col, self.market_col, self.price_col]
        if self.share_col:
            required.append(self.share_col)
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if self.share_col:
            group_shares = self.df.groupby(self.market_col)[self.share_col].sum()
            if (group_shares > 1.0 + 1e-6).any():
                raise ValueError("Some markets have shares summing above 1.")

        if (self.df[self.price_col] < 0).any():
            raise ValueError("Negative prices detected.")

    @property
    def firms(self) -> list[str]:
        return list(self.df[self.firm_col].unique())

    @property
    def markets(self) -> np.ndarray:
        return self.df[self.market_col].unique()

    @property
    def n_markets(self) -> int:
        return len(self.markets)

    def market_data(self, market_id) -> pd.DataFrame:
        return self.df[self.df[self.market_col] == market_id]

    def outside_shares(self) -> pd.Series:
        """Compute outside-good share per market: s_0 = 1 - sum(s_j)."""
        return 1.0 - self.df.groupby(self.market_col)[self.share_col].sum()

    def add_column(self, name: str, values) -> None:
        self.df[name] = values
