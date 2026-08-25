"""Implied volatility: invert a Black-Scholes price to recover sigma.

Given a market price, find the volatility that makes the Black-Scholes
formula reproduce it. Uses Newton-Raphson with the analytic vega as the
derivative, starting from a Brenner-Subrahmanyam guess, and falls back to
Brent's method (which is guaranteed a sign change inside its bracket) when
Newton struggles (e.g. for deep in/out-of-the-money options where vega is
tiny).

This module prices one option at a time (scalar inputs). The SVI module
(:mod:`engine.vol.svi`) will call it for each strike to build a surface.
"""

from __future__ import annotations

import numpy as np

from engine.core import (
    black_scholes_price,
    brent_root,
    european_lower_bound,
    newton_raphson,
    vega,
)
from engine.core.validation import OptionType, validate_pricing_inputs

# --- Volatility search bounds ---

#: Annualised volatility is assumed to live inside [MIN_VOL, MAX_VOL).
#: MIN_VOL is where the option price is (numerically) its lower bound and
#: MAX_VOL is large enough that the price has essentially reached its
#: no-arbitrage upper bound (S for a call, K*exp(-rT) for a put).
_MIN_VOL = 1e-8
_MAX_VOL = 50.0


# --- Initial guess ---

def _brenner_subrahmanyam_guess(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    option_type: str,
) -> float:
    """Initial guess for the implied volatility.

    Brenner-Subrahmanyam approximates the *at-the-money* call as:

        sigma ~ sqrt(2*pi/T) * (C / S)

    For puts we convert to the equivalent call price via put-call parity:

        C = P + S * exp(-qT) - K * exp(-rT)

    The guess is clipped into the search bounds so Newton always starts
    inside the valid volatility range.
    """
    call_price = market_price
    if option_type == "put":
        call_price = market_price + S * np.exp(-q * T) - K * np.exp(-r * T)

    guess = np.sqrt(2.0 * np.pi / T) * (call_price / S)
    return float(np.clip(guess, _MIN_VOL, _MAX_VOL))


# --- Implied volatility ---

def implied_volatility(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    market_price: float,
    option_type: OptionType,
    initial_guess: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Invert a European option's market price to find its implied volatility.

    Parameters:
        S: current underlying price
        K: strike price
        T: time to expiry in years (must be positive)
        r: continuously compounded risk-free rate
        q: continuous dividend yield (default 0)
        market_price: observed market price of the option
        option_type: either "call" or "put"
        initial_guess: starting volatility for Newton-Raphson
            (defaults to the Brenner-Subrahmanyam approximation)
        tolerance: how accurate the implied volatility must be
        max_iterations: maximum number of root-finding iterations

    Returns:
        The implied volatility as a float.

    Raises:
        ValueError: if the inputs are invalid, if the market price violates
            the no-arbitrage bounds (no volatility can reproduce it), or if
            neither root finder converges.
    """
    option_type = option_type.lower()

    # Reuse the shared checks; sigma is unknown here, so pass 0.0 (which the
    # pricing validation allows).
    validate_pricing_inputs(S=S, K=K, T=T, r=r, q=q, sigma=0.0, option_type=option_type)
    if T <= 0:
        raise ValueError("T must be positive when calculating implied volatility")
    if not np.isfinite(market_price):
        raise ValueError("market_price must be finite")
    if market_price < 0:
        raise ValueError("market_price cannot be negative")

    # No-arbitrage bounds: no finite volatility exists outside [lower, upper).
    # With dividends the call's ceiling is the discounted spot S*exp(-qT).
    lower_bound = european_lower_bound(S=S, K=K, T=T, r=r, q=q, option_type=option_type)
    upper_bound = float(S * np.exp(-q * T)) if option_type == "call" else float(K * np.exp(-r * T))

    if market_price < lower_bound:
        raise ValueError(
            f"market_price {market_price} is below the no-arbitrage lower bound "
            f"{lower_bound}; no volatility can reproduce it"
        )
    if market_price >= upper_bound:
        raise ValueError(
            f"market_price {market_price} is at or above the no-arbitrage upper "
            f"bound {upper_bound}; no finite volatility can reproduce it"
        )

    # At the lower bound the option is worth its intrinsic value: sigma = 0.
    if market_price - lower_bound <= 1e-12 * max(1.0, market_price):
        return 0.0

    # Root finding: f(sigma) = black_scholes_price(sigma) - market_price.
    # Volatility is clamped at MIN_VOL so a bad Newton step can never feed a
    # negative sigma into the pricer.
    def price_at(sigma: float) -> float:
        return black_scholes_price(
            S=S, K=K, T=T, r=r, q=q, sigma=float(max(sigma, _MIN_VOL)), option_type=option_type
        )

    function = lambda x: price_at(x) - market_price  # noqa: E731
    derivative = lambda x: vega(  # noqa: E731
        S=S, K=K, T=T, r=r, q=q, sigma=float(max(x, _MIN_VOL)), option_type=option_type
    )

    if initial_guess is not None and not (_MIN_VOL < initial_guess < _MAX_VOL):
        raise ValueError(f"initial_guess must be between {_MIN_VOL} and {_MAX_VOL}")

    guess = initial_guess if initial_guess is not None else _brenner_subrahmanyam_guess(
        S, K, T, r, q, market_price, option_type
    )

    result = newton_raphson(
        function=function,
        derivative=derivative,
        initial_guess=guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if result["converged"]:
        return float(max(result["root"], 0.0))

    # Newton struggled (typically vega ~ 0 for deep ITM/OTM options): fall
    # back to Brent, which is guaranteed a sign change inside the bracket
    # because f(MIN_VOL) < 0 and f(MAX_VOL) > 0.
    brent = brent_root(
        function=function,
        lower_bound=_MIN_VOL,
        upper_bound=_MAX_VOL,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if brent["converged"]:
        return float(brent["root"])

    raise ValueError(
        f"could not find an implied volatility for market_price {market_price} "
        f"(Newton-Raphson and Brent both failed to converge)"
    )
