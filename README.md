# Plug-and-Play Merger Simulator

A Python package for Bertrand-Nash merger simulations with pluggable demand models. Estimate demand, recover marginal costs, simulate mergers, and compute welfare effects.

## Installation

```bash
pip install -e ".[dev]"
```

## Demand Models


| Model        | Class          | Data Format           | Use When                                      |
| ------------ | -------------- | --------------------- | --------------------------------------------- |
| Plain logit  | `Logit`        | Long (share-based)    | IIA is acceptable; simple baseline            |
| Nested logit | `NestedLogit`  | Long (share-based)    | Products group into nests                     |
| Linear       | `LinearDemand` | Wide (quantity-based) | `Q = a + B*P` system; few products per market |
| Log-log      | `LogLogDemand` | Wide (quantity-based) | Constant-elasticity; `log Q = a + B*log P`    |


## Package Structure

```
merger_simulator/
├── __init__.py
├── data.py               # Data loading, reshaping, instrument construction
├── demand/
│   ├── __init__.py
│   ├── base.py            # Abstract base class for demand models
│   ├── logit.py           # Plain logit (Berry inversion)
│   ├── nested_logit.py    # Nested logit with configurable nest structure
│   ├── linear.py          # Linear demand system (Q = α + βP)
│   └── log_log.py         # Log-log demand (logQ = α + β logP)
├── instruments.py         # Hausman, rival cost shifters, BLP-style instruments
├── estimation.py          # IV / 2SLS estimation engine
├── merger.py              # Ownership matrices, equilibrium solvers
├── costs.py               # Marginal cost recovery, cost functions, efficiency defenses for logit
├── entry.py               # Entry/exit modeling (fixed-cost thresholds)
├── welfare.py             # Consumer surplus, producer surplus, HHI
└── utils.py               # Helpers (convergence checks, matrix ops)
```

## Example Usage

Look into [example.ipynb](merger_simulator/example.ipynb) to see how to use the package to simulate a merger (with fictional data) using a nested logit demand model.

## Quick Start: Nested Logit (Share-Based)

```python
import pandas as pd
import merger_simulator as ms

df = pd.read_csv("fast_food.csv")

# Prep columns (firm dummies for fixed effects)
df["const"] = 1
df["is_tacobell"] = (df["firm"] == "TacoBell").astype(int)
df["is_sushihut"] = (df["firm"] == "SushiHut").astype(int)

# 1. Wrap data
data = ms.MergerData(
    df,
    firm_col="firm",
    market_col="market_id",
    price_col="price",
    share_col="share",
    market_size_col="market_size",
    cost_shifter_cols=["cost_shifter"],
    product_chars=["advertising"],
    firm_id_col="firm_id",
)

# 2. Choose demand model — nest structure is declared here, once
model = ms.NestedLogit(nests={"legacy": ["McDonalds", "TacoBell"], "new": ["SushiHut"]})
model.apply_nests(data)  # writes nest labels onto data so instruments can use them

# 3. Build instruments
iv = (
    ms.hausman(data, time_col="quarter_id")
    + ms.rival_cost_shifters(data)
    + ms.blp_instruments(data)
    + ms.within_nest_rival_cost(data)
)

# 4. Estimate demand
result = model.estimate(
    data, instruments=iv,
    endog_cols=["price", "_ln_within_nest_share"],
    exog_cols=["const", "advertising", "is_tacobell", "is_sushihut"],
)
print(result.summary())
print(f"alpha = {model.alpha:.4f}, sigma = {model.sigma:.4f}")

# 5. Recover marginal costs
ms.recover_marginal_costs(model, data)
print(data.df.groupby("firm")[["_mc", "_markup"]].mean())

# 6. Simulate merger
merger = ms.simulate_merger(model, data, merging_firms=("McDonalds", "TacoBell"))
print(merger.summary())

# 7. Welfare
welfare = ms.compute_welfare(merger, alpha=model.alpha, merging_firms=("McDonalds", "TacoBell"))
print(welfare)
```

## Quick Start: Log-Log (Quantity-Based)

```python
import merger_simulator as ms

# Wide-format data: one row per market, columns like price_AA, passengers_AA, etc.
firms = ["AA", "DL", "UA"]
model = ms.LogLogDemand(
    firms=firms,
    price_cols={f: f"price_{f}" for f in firms},
    quantity_cols={f: f"passengers_{f}" for f in firms},
)

# Estimate (pass Hausman or other instruments as a DataFrame)
model.estimate(data, instruments=iv_df, exog_cols=["const", "avg_pop"])

# Recover MC and simulate
mc = model.recover_marginal_costs(df)
result = model.simulate_merger(df, merging=("AA", "DL"), tol=0.01, damping=0.1)
print(result["price_changes"])
print(result["profit_changes"])
```

`LinearDemand` works the same way -- just swap the class name and use linear prices/quantities instead of logs.

## Instruments

Build instrument sets and combine them with `+`:

```python
iv = ms.hausman(data, time_col="quarter_id")          # same-firm prices in other markets
iv = iv + ms.rival_cost_shifters(data)                 # rivals' mean cost shifter
iv = iv + ms.blp_instruments(data)                     # num_firms + rival characteristics
iv = iv + ms.within_nest_rival_cost(data)              # within-nest rival cost (nested logit)
```

## Cost Function & Efficiency Defense

```python
cost_fn = ms.CostFunction()
cost_fn.estimate(data, quantity_col="_quantity", instruments=cost_iv)
print(cost_fn.params)  # {'const': ..., 'cost_shifter': ..., 'quantity': ...}
```

## Entry Analysis

```python
from merger_simulator import entry_probability, simulate_entry

# Single market
prob = entry_probability(variable_profit=50.0, mu=2.9, sigma=0.6)

# Across markets
entry = simulate_entry(
    model, data,
    non_entry_markets=[1, 2, 3],
    entrant_firm="SushiHut",
    entrant_delta_base=1.58,
    entrant_mc_base_fn=lambda mid: 2.0,
    fixed_cost_mu=2.9,
    fixed_cost_sigma=0.6,
)
print(f"Entry rate: {entry.entry_rate:.1%}")
print(entry.summary())
```

## Key Outputs

- `EstimationResult.summary()` -- coefficient table with standard errors and t-stats
- `model.elasticity_matrix(data, market_id)` -- own/cross price elasticities
- `model.diversion_ratios(data, market_id)` -- diversion ratio matrix
- `MergerResult.summary()` -- mean/median/min/max price changes by firm
- `MergerResult.price_changes` -- market-level price change DataFrame
- `WelfareSummary` -- CS change, PS change, HHI pre/post, price index change

## Tests

```bash
pytest tests/ -v
```

42 tests covering all demand models, instruments, merger simulation, entry, and welfare.