# Options Pricing Engine

A from-scratch options pricing engine built in Python with NumPy and SciPy.

The engine prices European options with the Black-Scholes formula, computes
the full set of Greeks, and provides the numerical building blocks (root
finders, finite differences, no-arbitrage bounds) that the rest of the
project will build on: implied volatility, volatility surfaces, market data,
and a PDE solver.

## Module map

| Module                               | Status  | Purpose                                                       |
|--------------------------------------|---------|---------------------------------------------------------------|
| `engine/core/black_scholes.py`       | done    | European option pricing, `d1`/`d2`                            |
| `engine/core/greeks.py`              | done    | Delta, Gamma, Vega, Theta, Rho                                |
| `engine/core/numerical_utils.py`     | done    | Newton-Raphson, Brent, finite differences, bounds             |
| `engine/core/validation.py`          | done    | Shared input validation (scalars and arrays, NaN/Inf checks)  |
| `engine/vol/implied_vol.py`          | done    | Implied volatility via Newton-Raphson + Brent fallback        |
| `engine/vol/svi.py`                  | done    | SVI smile fitting (bounded least squares)                     |
| `engine/market/models.py`            | done    | `OptionQuote` / `OptionChain` data structures                 |
| `engine/market/data_fetcher.py`      | done    | Fetches real chains from Yahoo Finance (yfinance)             |
| `engine/market/arbitrage_filters.py` | done    | Vertical, butterfly, parity, lower-bound checks               |
| `engine/risk/crank_nicolson.py`      | done    | Crank-Nicolson PDE solver (European + American)               |
| `engine/utils/validation.py`         | stub    | Future project-wide validation helpers                        |
| `tests/`                             | done    | pytest suite covering all modules                             |

## Quickstart

```bash
pip install -e ".[dev]"   # install the package (and pytest) from the project root
```

```python
from engine.core import black_scholes_price, all_greeks

price = black_scholes_price(
    S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call"
)
greeks = all_greeks(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call")
```

All market inputs accept scalars **or** numpy arrays, so a whole option chain
can be priced in one call. Arguments are keyword-only to prevent accidentally
swapping `S` and `K`.

## Notebook

`notebooks/validation_against_hull.ipynb` validates the engine against Hull's
reference values (prices, Greeks, parity, the Crank-Nicolson solver,
implied-vol round trips) and runs the full live pipeline: fetch a real
chain, filter it, fit an SVI smile, and plot it. It falls back to a
synthetic chain when offline. Open it with `jupyter notebook` or VS Code.

The notebook needs a few extra packages beyond the core engine
(`matplotlib`, `pandas`, `jupyter`); install them with:

```bash
pip install -e ".[notebook]"
```

## Conventions

- **Theta** is annual (per 1 year decrease in T); divide by 365 for daily theta.
- **Vega** is per 1.00 change in volatility (e.g. 20% -> 21%).
- **Rho** is per 1.00 change in interest rate (e.g. 5% -> 6%).

## Roadmap

- [x] Tests for the core (Hull reference values, parity, edge cases)
- [x] Implied volatility (`engine/vol/implied_vol.py`)
- [x] Option data structures (`engine/market/models.py`)
- [x] Market data fetcher (`engine/market/data_fetcher.py`)
- [x] Arbitrage filters (`engine/market/arbitrage_filters.py`)
- [x] Crank-Nicolson PDE solver (European + American)
- [x] SVI smile fitting (`engine/vol/svi.py`)
- [x] Dividends (`q`) across the engine

## Contributing

Pull requests are welcome. For larger changes, open an issue first to discuss
what you would like to change. Please make sure existing tests still pass:

```bash
pytest
```
