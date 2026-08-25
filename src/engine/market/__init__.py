"""Market data module: option data structures, fetching, and arbitrage filtering.

Real option chains are noisy: stale quotes, crossed markets and missing
fields are common. This module exists to get that data into a clean,
typed form (:class:`~engine.market.models.OptionChain`) and to flag quotes
that break no-arbitrage relationships before they reach the SVI fitter.
"""

from engine.market.arbitrage_filters import (
    Violation,
    check_all,
    check_butterfly,
    check_lower_bounds,
    check_put_call_parity,
    check_vertical_spread,
    remove_arbitrage_violations,
)
from engine.market.data_fetcher import fetch_expiries, fetch_option_chain
from engine.market.models import OptionChain, OptionQuote

__all__ = [
    "OptionQuote",
    "OptionChain",
    "fetch_option_chain",
    "fetch_expiries",
    "Violation",
    "check_all",
    "check_lower_bounds",
    "check_vertical_spread",
    "check_butterfly",
    "check_put_call_parity",
    "remove_arbitrage_violations",
]
