"""Tests for the numerical utilities: root finders, finite differences, bounds."""

import numpy as np
import pytest

from engine.core import (
    black_scholes_price,
    brent_root,
    central_difference_first_derivative,
    central_difference_second_derivative,
    european_lower_bound,
    newton_raphson,
    option_intrinsic_value,
    vega,
)

# --- Newton-Raphson ----------------------------------------------------------


def test_newton_finds_linear_root():
    result = newton_raphson(
        function=lambda x: 2 * x - 4,
        derivative=lambda x: 2.0,
        initial_guess=1.0,
    )
    assert result["converged"] is True
    assert result["root"] == pytest.approx(2.0, abs=1e-8)
    assert result["method"] == "newton_raphson"
    assert 1 <= result["iterations"] <= 100


def test_newton_recovers_implied_volatility():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    target = black_scholes_price(
        S=S, K=K, T=T, r=r, sigma=sigma, option_type="call"
    )
    function = lambda x: black_scholes_price(  # noqa: E731
        S=S, K=K, T=T, r=r, sigma=x, option_type="call"
    ) - target
    derivative = lambda x: vega(S=S, K=K, T=T, r=r, sigma=x, option_type="call")  # noqa: E731

    result = newton_raphson(
        function=function, derivative=derivative, initial_guess=0.3
    )
    assert result["converged"] is True
    assert result["root"] == pytest.approx(sigma, abs=1e-8)


def test_newton_flat_derivative_reports_failure():
    # x^2 + 1 has no real root; Newton lands on x=0 where the derivative is
    # zero, so it must stop cleanly instead of dividing by zero.
    result = newton_raphson(
        function=lambda x: x**2 + 1,
        derivative=lambda x: 2 * x,
        initial_guess=1.0,
    )
    assert result["converged"] is False


def test_newton_iteration_cap_reports_failure():
    result = newton_raphson(
        function=lambda x: x - 10.0,
        derivative=lambda x: 1.0,
        initial_guess=0.0,
        max_iterations=1,
    )
    assert result["converged"] is False
    assert result["iterations"] == 1


# --- Brent -------------------------------------------------------------------


def test_brent_finds_root():
    result = brent_root(
        function=lambda x: x - 3.0, lower_bound=0.0, upper_bound=10.0
    )
    assert result["converged"] is True
    assert result["root"] == pytest.approx(3.0, abs=1e-8)
    assert result["method"] == "brent"


def test_brent_recovers_implied_volatility():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    target = black_scholes_price(
        S=S, K=K, T=T, r=r, sigma=sigma, option_type="call"
    )
    function = lambda x: black_scholes_price(  # noqa: E731
        S=S, K=K, T=T, r=r, sigma=x, option_type="call"
    ) - target

    result = brent_root(function=function, lower_bound=0.01, upper_bound=1.0)
    assert result["converged"] is True
    assert result["root"] == pytest.approx(sigma, abs=1e-8)


def test_brent_unbracketed_root_reports_failure():
    # x^2 + 1 never crosses zero, so there is no sign change in the bracket.
    result = brent_root(
        function=lambda x: x**2 + 1.0, lower_bound=-5.0, upper_bound=5.0
    )
    assert result["converged"] is False
    assert result["root"] is None


def test_brent_iteration_cap_reports_failure():
    # scipy's brentq raises RuntimeError when it exhausts maxiter; the
    # wrapper must convert that into a clean failure report.
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    target = black_scholes_price(
        S=S, K=K, T=T, r=r, sigma=0.2, option_type="call"
    )
    function = lambda x: black_scholes_price(  # noqa: E731
        S=S, K=K, T=T, r=r, sigma=x, option_type="call"
    ) - target

    result = brent_root(function=function, lower_bound=0.01, upper_bound=1.0, max_iterations=1)
    assert result["converged"] is False


# --- Finite differences ------------------------------------------------------


def test_first_derivative_cubic():
    # d/dx x^3 at x=2 is 12.
    assert central_difference_first_derivative(lambda x: x**3, 2.0) == pytest.approx(12.0, rel=1e-6)


def test_first_derivative_scales_with_small_x():
    # d/dx x^2 at x=1e-6 is 2e-6. A fixed step of 1e-5 would be far too big.
    assert central_difference_first_derivative(lambda x: x**2, 1e-6) == pytest.approx(2e-6, rel=1e-6)


def test_first_derivative_scales_with_large_x():
    # d/dx x^2 at x=1e6 is 2e6.
    assert central_difference_first_derivative(lambda x: x**2, 1e6) == pytest.approx(2e6, rel=1e-6)


def test_second_derivative_quartic():
    # d^2/dx^2 x^4 at x=3 is 12*3^2 = 108.
    assert central_difference_second_derivative(lambda x: x**4, 3.0) == pytest.approx(108.0, rel=1e-4)


def test_explicit_step_size_still_works():
    assert central_difference_first_derivative(
        lambda x: x**3, 2.0, step_size=1e-5
    ) == pytest.approx(12.0, rel=1e-6)


# --- Intrinsic value and lower bounds ----------------------------------------


@pytest.mark.parametrize(
    "spot,strike,expected",
    [(110.0, 100.0, 10.0), (90.0, 100.0, 0.0), (100.0, 100.0, 0.0)],
)
def test_intrinsic_value_call(spot, strike, expected):
    assert option_intrinsic_value(
        S=spot, K=strike, option_type="call"
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spot,strike,expected",
    [(110.0, 100.0, 0.0), (90.0, 100.0, 10.0), (100.0, 100.0, 0.0)],
)
def test_intrinsic_value_put(spot, strike, expected):
    assert option_intrinsic_value(
        S=spot, K=strike, option_type="put"
    ) == pytest.approx(expected)


def test_intrinsic_value_invalid_type():
    with pytest.raises(ValueError):
        option_intrinsic_value(S=100.0, K=100.0, option_type="straddle")


def test_european_call_lower_bound():
    # S - K*exp(-rT) = 100 - 100*exp(-0.05).
    bound = european_lower_bound(
        S=100.0, K=100.0, T=1.0, r=0.05, option_type="call"
    )
    assert bound == pytest.approx(100.0 - 100.0 * np.exp(-0.05))


def test_european_lower_bound_floors_at_zero():
    assert european_lower_bound(
        S=90.0, K=100.0, T=1.0, r=0.05, option_type="call"
    ) == pytest.approx(0.0)


def test_european_put_lower_bound():
    # K*exp(-rT) - S = 100*exp(-0.05) - 90.
    bound = european_lower_bound(
        S=90.0, K=100.0, T=1.0, r=0.05, option_type="put"
    )
    assert bound == pytest.approx(100.0 * np.exp(-0.05) - 90.0)


def test_lower_bound_invalid_type():
    with pytest.raises(ValueError):
        european_lower_bound(
            S=100.0, K=100.0, T=1.0, r=0.05, option_type="straddle"
        )


def test_intrinsic_and_lower_bound_accept_arrays():
    spots = np.array([90.0, 100.0, 110.0])
    intrinsic = option_intrinsic_value(S=spots, K=100.0, option_type="call")
    assert np.array_equal(intrinsic, np.array([0.0, 0.0, 10.0]))

    bound = european_lower_bound(
        S=spots, K=100.0, T=1.0, r=0.05, option_type="put"
    )
    expected = np.maximum(100.0 * np.exp(-0.05) - spots, 0.0)
    assert np.allclose(bound, expected)


# --- Dividend yield (q) ------------------------------------------------------


def test_european_lower_bound_with_dividends():
    q = 0.03
    discounted_spot = 100.0 * np.exp(-q)
    discounted_strike = 100.0 * np.exp(-0.05)
    assert european_lower_bound(
        S=100.0, K=100.0, T=1.0, r=0.05, q=q, option_type="call"
    ) == pytest.approx(max(discounted_spot - discounted_strike, 0.0))
    assert european_lower_bound(
        S=100.0, K=100.0, T=1.0, r=0.05, q=q, option_type="put"
    ) == pytest.approx(max(discounted_strike - discounted_spot, 0.0))
