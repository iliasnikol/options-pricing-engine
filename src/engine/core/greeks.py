"""Black-Scholes Greeks for European options.

The Greeks measure how much the option price changes when one input moves:
delta (underlying price), gamma (how fast delta changes), vega (volatility),
theta (time) and rho (interest rate).

The ``d1``/``d2`` terms and input validation are imported from
:mod:`engine.core.black_scholes` and :mod:`engine.core.validation`, so this
module can never drift out of sync with the pricing formulas.

All formulas include the continuous dividend yield ``q`` (setting ``q = 0``
recovers the classic no-dividend case). Rho is the sensitivity to the
risk-free rate ``r`` and is unchanged by dividends.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from engine.core.black_scholes import d1, d2
from engine.core.validation import (
    OptionType,
    ScalarOrArray,
    validate_greek_inputs,
)


# --- Delta ---


def delta(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> ScalarOrArray:
    """Calculate Black-Scholes Delta.

    Delta measures how much the option price changes when the underlying price S changes.

    For a call:
        Delta = exp(-qT) * N(d1)

    For a put:
        Delta = exp(-qT) * (N(d1) - 1)
    """
    option_type = option_type.lower()
    validate_greek_inputs(S, K, T, r, q, sigma, option_type)

    d1_value = d1(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    dividend_discount = np.exp(-q * T)

    if option_type == "call":
        return dividend_discount * norm.cdf(d1_value)

    return dividend_discount * (norm.cdf(d1_value) - 1)


# --- Gamma ---


def gamma(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> ScalarOrArray:
    """Calculate Black-Scholes Gamma.

    Gamma measures how quickly Delta changes when the underlying price S changes.

    Gamma is the same for calls and puts in the standard Black-Scholes model.
    """
    option_type = option_type.lower()
    validate_greek_inputs(S, K, T, r, q, sigma, option_type)

    d1_value = d1(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

    return np.exp(-q * T) * norm.pdf(d1_value) / (S * sigma * np.sqrt(T))


# --- Vega ---


def vega(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> ScalarOrArray:
    """Calculate Black-Scholes Vega.

    Vega measures how much the option price changes when volatility sigma changes.

    This returns Vega per 1.00 change in volatility.
    Example:
        If Vega = 37.5, then a volatility increase from 20% to 21%
        changes the option price by roughly 0.375.
    """
    option_type = option_type.lower()
    validate_greek_inputs(S, K, T, r, q, sigma, option_type)

    d1_value = d1(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

    return S * np.exp(-q * T) * norm.pdf(d1_value) * np.sqrt(T)


# --- Theta ---


def theta(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> ScalarOrArray:
    """Calculate Black-Scholes Theta.

    Theta measures how much the option price changes as time passes.

    This returns annual Theta, meaning the change in option price per 1 year decrease in T.
    To convert to daily Theta, divide the result by 365.
    """
    option_type = option_type.lower()
    validate_greek_inputs(S, K, T, r, q, sigma, option_type)

    d1_value = d1(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    d2_value = d2(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

    dividend_discount = np.exp(-q * T)
    first_term = -(S * dividend_discount * norm.pdf(d1_value) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        second_term = (
            -r * K * np.exp(-r * T) * norm.cdf(d2_value)
            + q * S * dividend_discount * norm.cdf(d1_value)
        )
        return first_term + second_term

    second_term = (
        r * K * np.exp(-r * T) * norm.cdf(-d2_value)
        - q * S * dividend_discount * norm.cdf(-d1_value)
    )
    return first_term + second_term


# --- Rho ---


def rho(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> ScalarOrArray:
    """Calculate Black-Scholes Rho.

    Rho measures how much the option price changes when the risk-free rate r changes.

    This returns Rho per 1.00 change in interest rates.
    Example:
        If Rho = 53.2, then a rate increase from 5% to 6%
        changes the option price by roughly 0.532.

    Rho is the sensitivity to ``r`` and is unchanged by the dividend yield.
    """
    option_type = option_type.lower()
    validate_greek_inputs(S, K, T, r, q, sigma, option_type)

    d2_value = d2(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2_value)

    return -K * T * np.exp(-r * T) * norm.cdf(-d2_value)


# --- All Greeks together ---


def all_greeks(
    *,
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray = 0.0,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> dict[str, ScalarOrArray]:
    """Calculate all main Black-Scholes Greeks and return them in a dictionary."""
    option_type = option_type.lower()
    validate_greek_inputs(S, K, T, r, q, sigma, option_type)

    return {
        "delta": delta(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type),
        "gamma": gamma(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type),
        "vega": vega(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type),
        "theta": theta(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type),
        "rho": rho(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type),
    }
