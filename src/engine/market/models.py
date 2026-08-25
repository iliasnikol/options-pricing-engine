"""Market data structures: the contract between data, filtering, and fitting.

An :class:`OptionQuote` is one listed contract; an :class:`OptionChain` is
everything listed for one underlying on one expiry date. Every other module
in the market pipeline (the fetcher that produces them, the arbitrage
filters that clean them, and the volatility-surface fitter that consumes
them) speaks in these types, so they come first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from engine.core.validation import OptionType

# --- Option quote ---

@dataclass(frozen=True)
class OptionQuote:
    """A single option contract as quoted in the market.

    Parameters:
        symbol: underlying ticker
        expiry: expiration date
        strike: strike price
        option_type: either "call" or "put"
        bid: best bid price
        ask: best ask price
        last: last traded price, if available
        volume: traded volume, if available
        open_interest: open interest, if available
    """

    symbol: str
    expiry: date
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None

    @property
    def mid(self) -> float:
        """Midpoint of the bid-ask spread, used as the standard fair quote price."""
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        """Width of the bid-ask spread (ask minus bid)."""
        return self.ask - self.bid

    def time_to_expiry(self, as_of: date, day_count: int = 365) -> float:
        """Years from ``as_of`` to expiry, using a simple day-count convention."""
        return (self.expiry - as_of).days / day_count


# --- Option chain ---

@dataclass
class OptionChain:
    """All listed options for one underlying on one expiry date.

    Parameters:
        symbol: underlying ticker
        as_of: the valuation date (today), used to compute time to expiry
        spot: current price of the underlying
        expiry: the expiration date of this chain
        quotes: the listed contracts
    """

    symbol: str
    as_of: date
    spot: float
    expiry: date
    quotes: list[OptionQuote] = field(default_factory=list)

    def of_type(self, option_type: str) -> list[OptionQuote]:
        """All quotes of the given type ("call" or "put", case-insensitive)."""
        return [q for q in self.quotes if q.option_type == option_type.lower()]

    def calls(self) -> list[OptionQuote]:
        """All call quotes, sorted by strike."""
        return sorted(self.of_type("call"), key=lambda q: q.strike)

    def puts(self) -> list[OptionQuote]:
        """All put quotes, sorted by strike."""
        return sorted(self.of_type("put"), key=lambda q: q.strike)

    def strikes(self) -> list[float]:
        """All strikes present in the chain, sorted, deduplicated."""
        return sorted({q.strike for q in self.quotes})

    def at_strike(self, strike: float, option_type: str) -> OptionQuote | None:
        """The quote for a strike and type, or None if it is not listed."""
        for quote in self.quotes:
            if quote.strike == strike and quote.option_type == option_type.lower():
                return quote
        return None

    def mid_price(self, strike: float, option_type: str) -> float | None:
        """Mid price at a strike and type, or None if it is not listed."""
        quote = self.at_strike(strike, option_type)
        return quote.mid if quote is not None else None

    def time_to_expiry(self, day_count: int = 365) -> float:
        """Years from the valuation date to expiry."""
        return (self.expiry - self.as_of).days / day_count
