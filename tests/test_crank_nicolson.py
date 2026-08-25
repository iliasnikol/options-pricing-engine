"""Tests for the Crank-Nicolson finite-difference solver."""

import numpy as np
import pytest

from engine.core import black_scholes_price
from engine.risk import crank_nicolson_price

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def cn(**overrides):
    """Call the solver with the reference inputs plus any overrides."""
    params = dict(
        S=S, K=K, T=T, r=R, sigma=SIGMA, option_type="call", exercise="european"
    )
    params.update(overrides)
    return crank_nicolson_price(**params)


# --- European options match the closed-form Black-Scholes price -------------


@pytest.mark.parametrize(
    "spot,strike,time,vol",
    [
        (100.0, 100.0, 1.0, 0.2),
        (90.0, 100.0, 0.5, 0.3),
        (110.0, 100.0, 2.0, 0.15),
        (95.0, 105.0, 0.25, 0.4),
        (120.0, 100.0, 1.0, 0.1),
    ],
)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_matches_black_scholes(spot, strike, time, vol, option_type):
    expected = black_scholes_price(
        S=spot, K=strike, T=time, r=R, sigma=vol, option_type=option_type
    )
    got = crank_nicolson_price(
        S=spot, K=strike, T=time, r=R, sigma=vol,
        option_type=option_type, exercise="european",
    )
    assert got == pytest.approx(expected, rel=2e-3)


def test_european_put_call_parity():
    call = cn(option_type="call")
    put = cn(option_type="put")
    assert call - put == pytest.approx(S - K * np.exp(-R * T), rel=1e-2)


def test_grid_refinement_reduces_error():
    exact = black_scholes_price(
        S=S, K=K, T=T, r=R, sigma=SIGMA, option_type="call"
    )
    errors = []
    for n_steps, m_steps in [(100, 50), (200, 100), (400, 200), (800, 400)]:
        got = crank_nicolson_price(
            S=S, K=K, T=T, r=R, sigma=SIGMA, option_type="call",
            exercise="european", n_steps=n_steps, m_steps=m_steps,
        )
        errors.append(abs(got - exact))
    # Crank-Nicolson is second order: halving both step sizes cuts the error
    # by roughly a factor of four.
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[3] < 5e-4


# --- American options --------------------------------------------------------


def test_american_call_equals_european_without_dividends():
    # Without dividends it is never optimal to exercise a call early, so the
    # American price must equal the European price.
    assert cn(exercise="american") == pytest.approx(
        cn(exercise="european"), rel=1e-9
    )


def test_american_put_is_worth_more_than_european():
    for spot in (90.0, 100.0):
        european = cn(S=spot, option_type="put", exercise="european")
        american = cn(S=spot, option_type="put", exercise="american")
        assert american > european


def test_american_put_never_below_intrinsic_value():
    # A deep ITM American put can always be exercised immediately for K - S.
    value = cn(S=80.0, option_type="put", exercise="american")
    assert value >= 20.0 - 1e-6


def test_american_put_has_positive_early_exercise_premium():
    premium = (
        cn(S=90.0, option_type="put", exercise="american")
        - cn(S=90.0, option_type="put", exercise="european")
    )
    assert premium > 0.5


# --- Validation --------------------------------------------------------------


def test_zero_time_raises():
    with pytest.raises(ValueError):
        cn(T=0.0)


def test_zero_volatility_raises():
    with pytest.raises(ValueError):
        cn(sigma=0.0)


def test_bad_exercise_type_raises():
    with pytest.raises(ValueError):
        cn(exercise="bermudan")


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        cn(option_type="straddle")


def test_coarse_grid_raises():
    with pytest.raises(ValueError):
        cn(n_steps=5)


def test_negative_spot_raises():
    with pytest.raises(ValueError):
        cn(S=-1.0)


# --- Dividend yield (q) ------------------------------------------------------


def test_european_with_dividends_matches_black_scholes():
    q = 0.03
    for option_type in ("call", "put"):
        expected = black_scholes_price(
            S=S, K=K, T=T, r=R, q=q, sigma=SIGMA, option_type=option_type
        )
        got = crank_nicolson_price(
            S=S, K=K, T=T, r=R, q=q, sigma=SIGMA,
            option_type=option_type, exercise="european",
        )
        assert got == pytest.approx(expected, rel=2e-3)


def test_american_call_with_dividends_can_exceed_european():
    # With q > r, early exercise of a call can be optimal, so the American
    # price exceeds the European price (impossible without dividends).
    q = 0.10
    american = crank_nicolson_price(
        S=S, K=K, T=T, r=R, q=q, sigma=SIGMA, option_type="call",
        exercise="american",
    )
    european = crank_nicolson_price(
        S=S, K=K, T=T, r=R, q=q, sigma=SIGMA, option_type="call",
        exercise="european",
    )
    assert american > european
    assert american - european > 0.5
