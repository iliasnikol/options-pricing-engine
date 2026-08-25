"""Tests for the static no-arbitrage filters."""

from datetime import date

import numpy as np
import pytest

from engine.market import (
    OptionChain,
    OptionQuote,
    check_all,
    check_butterfly,
    check_lower_bounds,
    check_put_call_parity,
    check_vertical_spread,
    remove_arbitrage_violations,
)

AS_OF = date(2025, 1, 16)
EXPIRY = date(2026, 1, 16)  # exactly one year
R = 0.05


def quote(strike, option_type, mid):
    """Zero-spread quote so mid prices are exact."""
    return OptionQuote(
        symbol="TEST", expiry=EXPIRY, strike=strike,
        option_type=option_type, bid=mid, ask=mid,
    )


def make_chain(spot, calls, puts):
    """Build a chain from {strike: mid} dicts for calls and puts."""
    quotes = [quote(k, "call", m) for k, m in calls.items()]
    quotes += [quote(k, "put", m) for k, m in puts.items()]
    return OptionChain(
        symbol="TEST", as_of=AS_OF, spot=spot, expiry=EXPIRY, quotes=quotes,
    )


def clean_chain():
    """A chain with no arbitrage violations (checks pass at tolerance 1e-9).

    Call mids are convex and fall with strike; put mids come from put-call
    parity, floored at the intrinsic value so the lower-bound check passes
    too (parity alone gives a European price that can sit below an American
    put's intrinsic value).
    """
    spot = 100.0
    calls = {95.0: 12.0, 100.0: 8.0, 105.0: 5.13}
    puts = {}
    for strike, call_mid in calls.items():
        parity = spot - strike * np.exp(-R * 1.0)
        intrinsic = max(strike - spot, 0.0)
        puts[strike] = max(call_mid - parity, intrinsic)
    return make_chain(spot, calls, puts)


# --- Clean chain passes every check -----------------------------------------


def test_clean_chain_has_no_violations():
    violations = check_all(clean_chain(), r=R, tolerance=1e-9)
    assert violations == []


# --- Lower bound -------------------------------------------------------------


def test_lower_bound_violation_for_itm_put():
    chain = make_chain(100.0, {}, {105.0: 4.0})  # intrinsic put = 5
    violations = check_lower_bounds(chain)
    assert len(violations) == 1
    assert violations[0].rule == "lower_bound"
    assert violations[0].excess == pytest.approx(1.0)


def test_lower_bound_violation_for_itm_call():
    chain = make_chain(100.0, {95.0: 3.0}, {})  # intrinsic call = 5
    violations = check_lower_bounds(chain)
    assert len(violations) == 1
    assert violations[0].excess == pytest.approx(2.0)


def test_lower_bound_respects_tolerance():
    chain = make_chain(100.0, {}, {105.0: 4.9})  # intrinsic put = 5, excess 0.1
    assert check_lower_bounds(chain, tolerance=0.1) == []
    assert len(check_lower_bounds(chain, tolerance=0.05)) == 1


# --- Vertical spread ---------------------------------------------------------


def test_call_vertical_spread_violation():
    chain = make_chain(100.0, {95.0: 5.0, 100.0: 6.0}, {})
    violations = check_vertical_spread(chain)
    assert len(violations) == 1
    assert violations[0].rule == "vertical_spread"
    assert violations[0].excess == pytest.approx(1.0)


def test_put_vertical_spread_violation():
    chain = make_chain(100.0, {}, {95.0: 6.0, 100.0: 5.0})
    violations = check_vertical_spread(chain)
    assert len(violations) == 1
    assert violations[0].excess == pytest.approx(1.0)


def test_vertical_spread_respects_tolerance():
    chain = make_chain(100.0, {95.0: 5.0, 100.0: 5.08}, {})
    assert check_vertical_spread(chain, tolerance=0.1) == []
    assert len(check_vertical_spread(chain, tolerance=0.05)) == 1


# --- Butterfly ---------------------------------------------------------------


def test_butterfly_violation():
    # C(100) = 8 sits above the interpolation of C(95)=5 and C(105)=2 (= 3.5).
    chain = make_chain(100.0, {95.0: 5.0, 100.0: 8.0, 105.0: 2.0}, {})
    violations = check_butterfly(chain)
    assert len(violations) == 1
    assert violations[0].rule == "butterfly"
    assert violations[0].excess == pytest.approx(4.5)


def test_butterfly_respects_tolerance():
    chain = make_chain(100.0, {95.0: 5.0, 100.0: 3.6, 105.0: 2.0}, {})
    # interpolation is 3.5, so excess is ~0.1
    assert check_butterfly(chain, tolerance=0.11) == []
    assert len(check_butterfly(chain, tolerance=0.05)) == 1


def test_butterfly_needs_three_strikes():
    chain = make_chain(100.0, {95.0: 5.0, 100.0: 8.0}, {})
    assert check_butterfly(chain) == []


# --- Put-call parity ---------------------------------------------------------


def test_parity_violation():
    # C - P = 8 - 3 = 5, parity expects S - K*exp(-rT) = 4.877...
    chain = make_chain(100.0, {100.0: 8.0}, {100.0: 3.0})
    violations = check_put_call_parity(chain, r=R)
    assert len(violations) == 1
    assert violations[0].rule == "put_call_parity"
    assert violations[0].excess == pytest.approx(0.1229, abs=1e-4)


def test_parity_skips_strikes_without_both_types():
    chain = make_chain(100.0, {100.0: 8.0}, {})
    assert check_put_call_parity(chain, r=R) == []


# --- check_all and removal ---------------------------------------------------


def test_check_all_combines_rules():
    chain = make_chain(100.0, {95.0: 5.0, 100.0: 6.0}, {105.0: 4.0})
    violations = check_all(chain, r=R)
    rules = {v.rule for v in violations}
    assert "vertical_spread" in rules
    assert "lower_bound" in rules


def test_remove_arbitrage_violations_drops_bad_quotes():
    good = quote(100.0, "call", 8.0)
    bad = quote(105.0, "put", 4.0)  # below intrinsic put = 5
    chain = OptionChain(
        symbol="TEST", as_of=AS_OF, spot=100.0, expiry=EXPIRY,
        quotes=[good, bad],
    )
    cleaned = remove_arbitrage_violations(chain, r=R)
    assert cleaned is not chain
    assert cleaned.quotes == [good]
    assert len(cleaned.quotes) == 1


# --- Dividend yield (q) ------------------------------------------------------


def test_parity_check_uses_dividend_yield():
    q = 0.03
    spot = 100.0
    strike = 100.0
    parity = spot * np.exp(-q * 1.0) - strike * np.exp(-R * 1.0)
    call_mid = 8.0
    put_mid = call_mid - parity
    chain = make_chain(spot, {strike: call_mid}, {strike: put_mid})
    # The mids satisfy parity only once dividends are taken into account.
    assert check_put_call_parity(chain, r=R, q=q) == []
    assert len(check_put_call_parity(chain, r=R)) == 1  # wrong without q
