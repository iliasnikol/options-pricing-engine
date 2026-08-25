"""Core pricing engine: Black-Scholes prices, Greeks, and numerical utilities.

Import the public API from here instead of the individual modules::

    from engine.core import black_scholes_price, all_greeks
"""

# --- Black-Scholes pricing ---
from engine.core.black_scholes import black_scholes_price, d1, d2

# --- Greeks ---
from engine.core.greeks import all_greeks, delta, gamma, rho, theta, vega

# --- Numerical utilities ---
from engine.core.numerical_utils import (
    brent_root,
    central_difference_first_derivative,
    central_difference_second_derivative,
    european_lower_bound,
    newton_raphson,
    option_intrinsic_value,
)

# --- Validation ---
from engine.core.validation import (
    OptionType,
    validate_greek_inputs,
    validate_pricing_inputs,
)

# --- Public API ---
__all__ = [
    "black_scholes_price",
    "d1",
    "d2",
    "all_greeks",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "newton_raphson",
    "brent_root",
    "central_difference_first_derivative",
    "central_difference_second_derivative",
    "option_intrinsic_value",
    "european_lower_bound",
    "OptionType",
    "validate_pricing_inputs",
    "validate_greek_inputs",
]
