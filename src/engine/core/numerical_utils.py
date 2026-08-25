"""Numerical utility functions for option pricing and Greeks.

These are the building blocks used by the rest of the engine: implied
volatility inverts option prices with the root finders, and the finite
differences cross-check the analytic Greeks.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

from engine.core.validation import OptionType, ScalarOrArray

# --- Newton-Raphson root finding ---


def newton_raphson(
    *,
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """
    Find a root of a function using the Newton-Raphson method.

    A root means a value x where:

        function(x) = 0

    Newton-Raphson starts with an initial guess and repeatedly updates it using:

        x_new = x_old - function(x_old) / derivative(x_old)

    Parameters:
        function:
            The function whose root we want to find.

        derivative:
            The derivative of that function.

        initial_guess:
            The starting value for the search.

        tolerance:
            How close to zero function(x) must be before we accept the answer.

        max_iterations:
            Maximum number of update steps before giving up.

    Returns:
        A dictionary with:
            root: the estimated root
            iterations: how many iterations were used
            converged: whether the method succeeded
            method: "newton_raphson"
    """

    x = initial_guess

    for iteration in range(1, max_iterations + 1):
        function_value = function(x)
        derivative_value = derivative(x)

        # If function(x) is already close enough to zero, we are done.
        if abs(function_value) < tolerance:
            return {
                "root": x,
                "iterations": iteration,
                "converged": True,
                "method": "newton_raphson",
            }

        # If the derivative is too close to zero, the Newton step would explode.
        # This is especially important for implied volatility, where vega can be tiny.
        if abs(derivative_value) < 1e-12:
            return {
                "root": x,
                "iterations": iteration,
                "converged": False,
                "method": "newton_raphson",
            }

        x_next = x - (function_value / derivative_value)

        # If the update barely changes x, we treat that as convergence.
        if abs(x_next - x) < tolerance:
            return {
                "root": x_next,
                "iterations": iteration,
                "converged": True,
                "method": "newton_raphson",
            }

        x = x_next

    return {
        "root": x,
        "iterations": max_iterations,
        "converged": False,
        "method": "newton_raphson",
    }

# --- Brent root finding ---


def brent_root(
    *,
    function: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """
    Find a root of a function using Brent's method.

    Brent's method is more reliable than Newton-Raphson because it searches inside
    a bracket:

        [lower_bound, upper_bound]

    The key requirement is that the function must change sign across the bracket:

        function(lower_bound) and function(upper_bound)

    must have opposite signs.

    Parameters:
        function:
            The function whose root we want to find.

        lower_bound:
            Lower end of the search interval.

        upper_bound:
            Upper end of the search interval.

        tolerance:
            How accurate the root should be.

        max_iterations:
            Maximum number of iterations.

    Returns:
        A dictionary with:
            root: the estimated root
            iterations: how many iterations were used
            converged: whether the method succeeded
            method: "brent"
    """

    try:
        root, result = brentq(
            function,
            lower_bound,
            upper_bound,
            xtol=tolerance,
            maxiter=max_iterations,
            full_output=True,
        )

        return {
            "root": root,
            "iterations": result.iterations,
            "converged": result.converged,
            "method": "brent",
        }

    except (ValueError, RuntimeError):
        # ValueError: the root is not bracketed, e.g. function(lower_bound)
        # and function(upper_bound) are both positive.
        # RuntimeError: brentq ran out of iterations before converging.
        # In both cases we report failure instead of crashing the caller.
        return {
            "root": None,
            "iterations": 0,
            "converged": False,
            "method": "brent",
        }

# --- Finite differences ---


def _scale_aware_step(x: float, exponent: float) -> float:
    """Pick a finite-difference step that scales with |x|.

    A fixed step (e.g. 1e-5) is a poor fit when x is very small or very
    large: too small a step drowns in floating-point roundoff, too large a
    step picks up truncation error. The standard compromise is
    eps^(1/exponent) * max(1, |x|), which balances both at any scale.
    """
    return float(np.finfo(float).eps ** (1.0 / exponent) * max(1.0, abs(x)))


def central_difference_first_derivative(
    function: Callable[[float], float],
    x: float,
    step_size: float | None = None,
) -> float:
    """
    Approximate the first derivative of a function using a central difference.

    The first derivative measures the slope of the function.

    Instead of using:

        f'(x) ≈ [f(x + h) - f(x)] / h

    we use the more accurate central version:

        f'(x) ≈ [f(x + h) - f(x - h)] / (2h)

    Parameters:
        function:
            The function we want to differentiate.

        x:
            The point where we want the derivative.

        step_size:
            The small movement h around x. Defaults to a step that scales
            with |x| (pass an explicit value for the old fixed-step behaviour).

    Returns:
        Approximate first derivative at x.
    """

    h = _scale_aware_step(x, 3) if step_size is None else step_size

    return (function(x + h) - function(x - h)) / (2 * h)


def central_difference_second_derivative(
    function: Callable[[float], float],
    x: float,
    step_size: float | None = None,
) -> float:
    """
    Approximate the second derivative of a function using a central difference.

    The second derivative measures curvature.

    This is useful for Gamma, because Gamma is the second derivative of option
    price with respect to the underlying price S.

    Formula:

        f''(x) ≈ [f(x + h) - 2f(x) + f(x - h)] / h²

    Parameters:
        function:
            The function we want to differentiate.

        x:
            The point where we want the second derivative.

        step_size:
            The small movement h around x. Defaults to a step that scales
            with |x| (pass an explicit value for the old fixed-step behaviour).

    Returns:
        Approximate second derivative at x.
    """

    h = _scale_aware_step(x, 4) if step_size is None else step_size

    return (function(x + h) - 2 * function(x) + function(x - h)) / h**2

# --- Intrinsic value and no-arbitrage bounds ---


def option_intrinsic_value(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    option_type: OptionType,
) -> ScalarOrArray:
    """
    Calculate the intrinsic value of a European option at expiry.

    Intrinsic value means the immediate payoff if the option expires now.

    For a call:

        max(S - K, 0)

    For a put:

        max(K - S, 0)

    Parameters:
        S:
            Current underlying price.

        K:
            Strike price.

        option_type:
            Either "call" or "put".

    Returns:
        The option's intrinsic value.
    """

    option_type = option_type.lower()

    if option_type == "call":
        return np.maximum(S - K, 0.0)

    if option_type == "put":
        return np.maximum(K - S, 0.0)

    raise ValueError("option_type must be either 'call' or 'put'")


def european_lower_bound(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    option_type: OptionType,
) -> ScalarOrArray:
    """
    Calculate the European no-arbitrage lower bound.

    This is not always the same as intrinsic value before expiry.

    For a European call with continuous dividend yield q:

        lower bound = max(S * exp(-qT) - K * exp(-rT), 0)

    For a European put with continuous dividend yield q:

        lower bound = max(K * exp(-rT) - S * exp(-qT), 0)

    Why discount K and S?
        A European option can only be exercised at expiry. The strike payment
        happens in the future, so its present value is K * exp(-rT), and the
        stock pays out dividends before expiry, so its future value today is
        S * exp(-qT). Setting q = 0 recovers the classic no-dividend bounds.

    This is different from an American option, where early exercise may be allowed.
    That is one of the known bugs this file is meant to avoid.
    """

    option_type = option_type.lower()

    discounted_strike = K * np.exp(-r * T)
    discounted_spot = S * np.exp(-q * T)

    if option_type == "call":
        return np.maximum(discounted_spot - discounted_strike, 0.0)

    if option_type == "put":
        return np.maximum(discounted_strike - discounted_spot, 0.0)

    raise ValueError("option_type must be either 'call' or 'put'")
