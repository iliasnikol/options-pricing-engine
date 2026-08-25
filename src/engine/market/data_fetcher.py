"""Fetch real option chains from Yahoo Finance (via yfinance).

The network calls are deliberately thin: ``fetch_option_chain`` downloads
the two tables (calls and puts) and hands them to
:func:`_quotes_from_frame`, a pure conversion function that turns a row of
market data into an :class:`~engine.market.models.OptionQuote`. Keeping the
conversion pure means it can be unit-tested without touching the network.

yfinance is imported lazily inside the fetch functions so that importing
``engine.market`` never requires it (and tests that do not hit the network
stay fast).
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np

from engine.market.models import OptionChain, OptionQuote

# --- Field conversion helpers ---

def _float_or_none(value) -> float | None:
    """Convert a market-data field to float, mapping NaN/None to None."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def _float_or_zero(value) -> float:
    """Convert a market-data field to float, mapping NaN/None to 0.0."""
    value = _float_or_none(value)
    return 0.0 if value is None else value


def _int_or_none(value) -> int | None:
    """Convert a market-data field to int, mapping NaN/None to None."""
    value = _float_or_none(value)
    return None if value is None else int(value)


def _quotes_from_frame(
    frame,
    symbol: str,
    expiry: date,
    option_type: str,
) -> list[OptionQuote]:
    """Convert a yfinance calls/puts DataFrame into OptionQuote objects.

    ``frame`` is a pandas DataFrame with the standard yfinance columns
    (strike, bid, ask, lastPrice, volume, openInterest); missing bid/ask
    values become 0.0, other missing fields become None.
    """
    quotes = []
    for row in frame.to_dict("records"):
        quotes.append(
            OptionQuote(
                symbol=symbol,
                expiry=expiry,
                strike=float(row["strike"]),
                option_type=option_type,
                bid=_float_or_zero(row.get("bid")),
                ask=_float_or_zero(row.get("ask")),
                last=_float_or_none(row.get("lastPrice")),
                volume=_int_or_none(row.get("volume")),
                open_interest=_int_or_none(row.get("openInterest")),
            )
        )
    return quotes


# --- Fetching from Yahoo Finance ---

def _spot_price(ticker) -> float:
    """Best-effort spot price from a yfinance Ticker object."""
    try:
        return float(ticker.fast_info["lastPrice"])
    except Exception:
        return float(ticker.info["regularMarketPrice"])


def fetch_expiries(ticker: str) -> list[date]:
    """List the expiration dates currently listed for a ticker."""
    import yfinance as yf

    return [
        datetime.strptime(expiry, "%Y-%m-%d").date()
        for expiry in yf.Ticker(ticker).options
    ]


def fetch_option_chain(
    ticker: str,
    expiry: str,
    as_of: date | None = None,
) -> OptionChain:
    """Fetch the option chain for ``ticker`` at the ``expiry`` date.

    Parameters:
        ticker: the underlying ticker, e.g. "AAPL"
        expiry: the expiration date as a "YYYY-MM-DD" string
            (see :func:`fetch_expiries` for what is available)
        as_of: the valuation date; defaults to today

    Returns:
        An :class:`~engine.market.models.OptionChain` for that expiry.

    Raises:
        ImportError: if yfinance is not installed.
        Exception: whatever yfinance raises when the ticker/expiry is invalid
            (e.g. an expiry that is not listed).
    """
    import yfinance as yf

    if as_of is None:
        as_of = date.today()

    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    data = yf.Ticker(ticker)
    chain = data.option_chain(expiry)

    quotes = _quotes_from_frame(chain.calls, ticker, expiry_date, "call")
    quotes += _quotes_from_frame(chain.puts, ticker, expiry_date, "put")

    return OptionChain(
        symbol=ticker,
        as_of=as_of,
        spot=_spot_price(data),
        expiry=expiry_date,
        quotes=quotes,
    )
