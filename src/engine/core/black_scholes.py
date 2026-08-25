"""Black-Scholes pricing for European options.

Implements the closed-form Black-Scholes formula (including a continuous
dividend yield ``q``), plus the ``d1``/``d2`` terms that the Greeks module
(:mod:`engine.core.greeks`) also needs, so the two can never drift out of
sync.

With dividends the formula simply replaces ``S`` by ``S * exp(-qT)`` in the
stock-price terms and ``(r + 0.5*sigma^2)`` by ``(r - q + 0.5*sigma^2)`` in
d1; setting ``q = 0`` recovers the classic formula.

All market inputs accept scalars or numpy arrays; validation lives in
:mod:`engine.core.validation`. Arguments are keyword-only to make it
impossible to silently swap two inputs (e.g. ``S`` and ``K``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from engine.core.validation import (
    OptionType,
    ScalarOrArray,
    validate_pricing_inputs,
)

# --- Calculate d1 ---


def d1(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Calculate d1 from the Black-Scholes formula.

    d1 appears in Delta and in the stock-price-sensitive part of the option price.
    """
    numerator = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
    denominator = sigma * np.sqrt(T)

    return numerator / denominator

# --- Calculate d2 ---


def d2(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Calculate d2 from d1.

    d2 is d1 shifted down by one volatility-adjusted time step.
    """
    return d1(S=S, K=K, T=T, r=r, q=q, sigma=sigma) - sigma * np.sqrt(T)

# --- Price a European option with the Black-Scholes formula ---


def black_scholes_price(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> ScalarOrArray:
    """Price a European option using the Black-Scholes formula.

    Parameters:
        S: current underlying price
        K: strike price
        T: time to expiry in years
        r: continuously compounded risk-free rate
        q: continuous dividend yield (default 0)
        sigma: annualised volatility
        option_type: either "call" or "put"

    All parameters may be scalars or numpy arrays (validation checks that
    every element is valid). ``T == 0`` and ``sigma == 0`` are handled
    analytically via the intrinsic / forward value.
    """
    option_type = option_type.lower()

    # Validate inputs
    validate_pricing_inputs(S, K, T, r, q, sigma, option_type)

    T = np.asarray(T)
    sigma = np.asarray(sigma)

    # The formula divides by sigma * sqrt(T); where either is zero we fall
    # back to the degenerate (intrinsic / forward) value below, so suppress
    # the benign divide-by-zero warnings here.
    with np.errstate(divide="ignore", invalid="ignore"):
        d1_value = d1(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        d2_value = d2(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

        discounted_spot = S * np.exp(-q * T)
        discounted_strike = K * np.exp(-r * T)

        if option_type == "call":
            price = discounted_spot * norm.cdf(d1_value) - discounted_strike * norm.cdf(d2_value)
        else:
            price = discounted_strike * norm.cdf(-d2_value) - discounted_spot * norm.cdf(-d1_value)

    # Handle edge cases where T or sigma is zero: the price degenerates to
    # the intrinsic value with spot and strike discounted back to today.
    degenerate = (T == 0) | (sigma == 0)
    if np.any(degenerate):
        discounted_spot = S * np.exp(-q * T)
        discounted_strike = K * np.exp(-r * T)
        if option_type == "call":
            intrinsic = np.maximum(discounted_spot - discounted_strike, 0.0)
        else:
            intrinsic = np.maximum(discounted_strike - discounted_spot, 0.0)
        price = np.where(degenerate, intrinsic, price)

    return price
