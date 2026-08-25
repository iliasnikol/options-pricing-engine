"""Tests for the Black-Scholes European option pricer."""

import numpy as np
import pytest

from engine.core import black_scholes_price, d1, d2, european_lower_bound

# Reference values from Hull, "Options, Futures, and Other Derivatives",
# for S=100, K=100, T=1 year, r=5%, sigma=20%.
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
HULL_CALL = 10.450583572185565
HULL_PUT = 5.573526022256971


def price(**overrides):
    """Call the pricer with the Hull reference inputs plus any overrides."""
    params = dict(S=S, K=K, T=T, r=R, sigma=SIGMA, option_type="call")
    params.update(overrides)
    return black_scholes_price(**params)


# --- Hull reference values --------------------------------------------------


def test_hull_call_price():
    assert price(option_type="call") == pytest.approx(HULL_CALL, rel=1e-9)


def test_hull_put_price():
    assert price(option_type="put") == pytest.approx(HULL_PUT, rel=1e-9)


# --- No-arbitrage relationships ---------------------------------------------


def test_put_call_parity():
    call = price(option_type="call")
    put = price(option_type="put")
    assert call - put == pytest.approx(S - K * np.exp(-R * T), rel=1e-12)


def test_price_stays_between_lower_bound_and_underlying():
    for spot in (90.0, 100.0, 110.0):
        call = price(S=spot, option_type="call")
        put = price(S=spot, option_type="put")
        bound = european_lower_bound(S=spot, K=K, T=T, r=R, option_type="call")
        assert call >= bound - 1e-12
        assert call < spot
        assert 0.0 < put < K


def test_deep_itm_call_approaches_forward_value():
    # For a very deep ITM call, C -> S - K * exp(-rT).
    expected = 1e6 - K * np.exp(-R * T)
    assert price(S=1e6, option_type="call") == pytest.approx(expected, rel=1e-9)


def test_deep_otm_put_is_worthless():
    assert price(S=1e6, option_type="put") == pytest.approx(0.0, abs=1e-12)


def test_negative_rates_are_supported():
    # Negative interest rates are valid market data and must not be rejected.
    assert np.isfinite(price(r=-0.01))


# --- Edge cases: T = 0 and sigma = 0 ----------------------------------------


def test_expiry_price_is_payoff():
    # At expiry the option is worth exactly its intrinsic value.
    assert price(S=90.0, T=0.0, option_type="call") == pytest.approx(0.0)
    assert price(S=110.0, T=0.0, option_type="call") == pytest.approx(10.0)
    assert price(S=90.0, T=0.0, option_type="put") == pytest.approx(10.0)
    assert price(S=110.0, T=0.0, option_type="put") == pytest.approx(0.0)


def test_zero_volatility_price_is_discounted_intrinsic():
    # With sigma = 0 the option is worth max(S - K*exp(-rT), 0).
    discounted_strike = K * np.exp(-R * T)
    assert price(S=110.0, sigma=0.0, option_type="call") == pytest.approx(110.0 - discounted_strike)
    assert price(S=90.0, sigma=0.0, option_type="call") == pytest.approx(0.0)
    assert price(S=90.0, sigma=0.0, option_type="put") == pytest.approx(discounted_strike - 90.0)


# --- Input validation --------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        dict(S=0.0),
        dict(S=-5.0),
        dict(K=0.0),
        dict(K=-5.0),
        dict(T=-0.5),
        dict(sigma=-0.1),
        dict(option_type="clal"),
        dict(option_type=""),
    ],
)
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        price(**bad)


@pytest.mark.parametrize(
    "bad",
    [
        dict(S=float("nan")),
        dict(S=float("inf")),
        dict(K=float("nan")),
        dict(T=float("inf")),
        dict(r=float("nan")),
        dict(sigma=float("nan")),
        dict(sigma=float("-inf")),
    ],
)
def test_non_finite_inputs_raise(bad):
    with pytest.raises(ValueError):
        price(**bad)


def test_option_type_is_case_insensitive():
    assert price(option_type="CALL") == price(option_type="call")
    assert price(option_type="Put") == price(option_type="put")


def test_positional_arguments_rejected():
    # Keyword-only arguments make it impossible to swap S and K by accident.
    with pytest.raises(TypeError):
        black_scholes_price(S, K, T, R, SIGMA, "call")


# --- d1 / d2 -----------------------------------------------------------------


def test_d2_relationship_with_d1():
    d1_value = d1(S=S, K=K, T=T, r=R, sigma=SIGMA)
    d2_value = d2(S=S, K=K, T=T, r=R, sigma=SIGMA)
    assert d2_value == pytest.approx(d1_value - SIGMA * np.sqrt(T))


# --- Array support -----------------------------------------------------------


def test_array_prices_match_scalar():
    spots = np.array([90.0, 100.0, 110.0])
    prices = price(S=spots)
    for spot, value in zip(spots, prices):
        assert value == pytest.approx(price(S=float(spot)))


def test_mixed_arrays_with_degenerate_entries():
    spots = np.array([90.0, 100.0, 110.0])
    times = np.array([0.0, 1.0, 2.0])
    vols = np.array([0.0, 0.2, 0.3])
    prices = price(S=spots, T=times, sigma=vols)
    assert prices[0] == pytest.approx(0.0)  # T=0 and sigma=0, OTM call
    assert prices[1] == pytest.approx(HULL_CALL, rel=1e-9)
    assert prices[2] == pytest.approx(
        price(S=110.0, T=2.0, sigma=0.3), rel=1e-9
    )


def test_array_validation_raises_if_any_element_invalid():
    with pytest.raises(ValueError):
        price(S=np.array([90.0, -1.0, 110.0]))


# --- Dividend yield (q) ------------------------------------------------------


def test_put_call_parity_with_dividends():
    q = 0.03
    call = price(q=q, option_type="call")
    put = price(q=q, option_type="put")
    assert call - put == pytest.approx(
        S * np.exp(-q * T) - K * np.exp(-R * T), rel=1e-12
    )


def test_price_stays_above_lower_bound_with_dividends():
    q = 0.03
    for spot in (90.0, 100.0, 110.0):
        call = price(S=spot, q=q, option_type="call")
        bound = european_lower_bound(S=spot, K=K, T=T, r=R, q=q, option_type="call")
        assert call >= bound - 1e-12


def test_zero_volatility_with_dividends():
    q = 0.03
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-R * T)
    assert price(sigma=0.0, q=q, option_type="call") == pytest.approx(
        max(discounted_spot - discounted_strike, 0.0)
    )
    assert price(S=90.0, sigma=0.0, q=q, option_type="put") == pytest.approx(
        max(discounted_strike - 90.0 * np.exp(-q * T), 0.0)
    )


def test_dividend_yield_must_be_finite():
    with pytest.raises(ValueError):
        price(q=float("nan"))
