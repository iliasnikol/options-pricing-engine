"""Tests for the OptionQuote / OptionChain data structures."""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from engine.market import OptionChain, OptionQuote

EXPIRY = date(2026, 1, 16)
AS_OF = date(2025, 1, 16)


def quote(strike=100.0, option_type="call", bid=5.0, ask=5.0):
    return OptionQuote(
        symbol="TEST", expiry=EXPIRY, strike=strike,
        option_type=option_type, bid=bid, ask=ask,
    )


def chain(*quotes, spot=100.0):
    return OptionChain(
        symbol="TEST", as_of=AS_OF, spot=spot, expiry=EXPIRY,
        quotes=list(quotes),
    )


# --- OptionQuote -------------------------------------------------------------


def test_mid_is_average_of_bid_and_ask():
    q = quote(bid=4.0, ask=6.0)
    assert q.mid == pytest.approx(5.0)


def test_spread_is_ask_minus_bid():
    q = quote(bid=4.0, ask=6.0)
    assert q.spread == pytest.approx(2.0)


def test_time_to_expiry_full_year():
    q = quote()
    assert q.time_to_expiry(AS_OF) == pytest.approx(1.0)


def test_time_to_expiry_uses_day_count():
    # 2025-07-16 to 2026-01-16 is 184 days.
    as_of = date(2025, 7, 16)
    q = quote()
    assert q.time_to_expiry(as_of) == pytest.approx(184 / 365)


def test_quote_is_immutable():
    q = quote()
    with pytest.raises(FrozenInstanceError):
        q.bid = 10.0


# --- OptionChain -------------------------------------------------------------


def test_calls_sorted_by_strike():
    c = chain(
        quote(strike=105.0, option_type="call"),
        quote(strike=95.0, option_type="call"),
        quote(strike=100.0, option_type="call"),
    )
    assert [q.strike for q in c.calls()] == [95.0, 100.0, 105.0]


def test_puts_sorted_by_strike():
    c = chain(
        quote(strike=105.0, option_type="put"),
        quote(strike=95.0, option_type="put"),
    )
    assert [q.strike for q in c.puts()] == [95.0, 105.0]


def test_of_type_filters_and_ignores_case():
    c = chain(
        quote(strike=100.0, option_type="call"),
        quote(strike=100.0, option_type="put"),
    )
    assert len(c.of_type("CALL")) == 1
    assert len(c.of_type("Put")) == 1
    assert c.of_type("call")[0].option_type == "call"


def test_strikes_are_unique_and_sorted():
    c = chain(
        quote(strike=105.0), quote(strike=95.0),
        quote(strike=105.0, option_type="put"), quote(strike=95.0, option_type="put"),
    )
    assert c.strikes() == [95.0, 105.0]


def test_at_strike_finds_quote():
    c = chain(quote(strike=100.0), quote(strike=100.0, option_type="put"))
    found = c.at_strike(100.0, "call")
    assert found is not None and found.option_type == "call"
    assert c.at_strike(95.0, "call") is None


def test_mid_price_helper():
    c = chain(quote(strike=100.0, bid=4.0, ask=6.0))
    assert c.mid_price(100.0, "call") == pytest.approx(5.0)
    assert c.mid_price(100.0, "put") is None


def test_chain_time_to_expiry():
    c = chain(quote())
    assert c.time_to_expiry() == pytest.approx(1.0)
