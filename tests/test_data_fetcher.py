"""Offline tests for the market data fetcher.

The network calls in :mod:`engine.market.data_fetcher` are thin wrappers;
the interesting logic is the conversion from a yfinance DataFrame to
:class:`OptionQuote` objects, which is tested here with a hand-built
DataFrame that mimics what yfinance returns.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine.market.data_fetcher import (
    _float_or_none,
    _int_or_none,
    _quotes_from_frame,
)

EXPIRY = date(2026, 1, 16)


def calls_frame():
    """Mimic a yfinance calls table, including a row with missing data."""
    return pd.DataFrame(
        [
            {
                "strike": 95.0,
                "bid": 11.9,
                "ask": 12.1,
                "lastPrice": 12.0,
                "volume": 500,
                "openInterest": 1000,
            },
            {
                "strike": 100.0,
                "bid": float("nan"),
                "ask": float("nan"),
                "lastPrice": float("nan"),
                "volume": float("nan"),
                "openInterest": float("nan"),
            },
            {
                "strike": 105.0,
                "bid": 4.9,
                "ask": 5.1,
                "lastPrice": 5.05,
                "volume": 0,
                "openInterest": 250,
            },
        ]
    )


def test_converts_populated_row():
    quotes = _quotes_from_frame(calls_frame(), "TEST", EXPIRY, "call")
    assert len(quotes) == 3

    first = quotes[0]
    assert first.symbol == "TEST"
    assert first.expiry == EXPIRY
    assert first.option_type == "call"
    assert first.strike == pytest.approx(95.0)
    assert first.bid == pytest.approx(11.9)
    assert first.ask == pytest.approx(12.1)
    assert first.last == pytest.approx(12.0)
    assert first.volume == 500
    assert first.open_interest == 1000
    assert first.mid == pytest.approx(12.0)


def test_missing_fields_become_defaults():
    quotes = _quotes_from_frame(calls_frame(), "TEST", EXPIRY, "call")
    missing = quotes[1]
    # Missing bid/ask default to 0.0 so the quote is still usable (and the
    # arbitrage filters will flag it); other fields become None.
    assert missing.bid == 0.0
    assert missing.ask == 0.0
    assert missing.last is None
    assert missing.volume is None
    assert missing.open_interest is None


def test_zero_volume_is_preserved():
    quotes = _quotes_from_frame(calls_frame(), "TEST", EXPIRY, "call")
    assert quotes[2].volume == 0


def test_puts_get_put_type():
    quotes = _quotes_from_frame(calls_frame(), "TEST", EXPIRY, "put")
    assert all(q.option_type == "put" for q in quotes)


def test_float_or_none_maps_nan_to_none():
    assert _float_or_none(float("nan")) is None
    assert _float_or_none(None) is None
    assert _float_or_none(3.5) == 3.5


def test_int_or_none_maps_nan_to_none():
    assert _int_or_none(float("nan")) is None
    assert _int_or_none(3.7) == 3
    assert _int_or_none(0) == 0
