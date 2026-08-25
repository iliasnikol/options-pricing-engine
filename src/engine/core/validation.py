"""Shared input validation for the Black-Scholes pricers in :mod:`engine.core`.

The engine accepts both scalar and array inputs for the market parameters
``S``, ``K``, ``T``, ``r``, ``q`` and ``sigma``, so validation is done
elementwise with numpy: an error is raised if *any* element of any input is
invalid. NaN and infinite values are rejected explicitly (a plain
``if S <= 0`` check silently lets NaN through, because comparisons with NaN
are always False).

There are two flavours of validation:

- :func:`validate_pricing_inputs` allows ``T == 0`` and ``sigma == 0``
  (the pricing functions handle those degenerate cases analytically).
- :func:`validate_greek_inputs` additionally requires ``T > 0`` and
  ``sigma > 0``, because every Greek formula divides by ``sqrt(T)`` or
  ``sigma`` (at expiry the Greeks are discontinuous or undefined).
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

# --- Option types and shared type aliases ---

#: Allowed option types, checked at runtime and hinted for static checkers.
OptionType = Literal["call", "put"]

#: A single number or a numpy array (used for S, K, T, r, sigma).
ScalarOrArray = Union[float, np.ndarray]

_VALID_OPTION_TYPES = frozenset({"call", "put"})

# --- Shared finite-value check ---


def _require_finite(name: str, values: ScalarOrArray) -> None:
    """Raise if ``values`` contains any NaN or infinite entries."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

# --- Pricing input validation ---


def validate_pricing_inputs(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> None:
    """Validate inputs for the pricing functions.

    ``T`` and ``sigma`` may be zero here; the Black-Scholes price handles
    those cases analytically. ``q`` is the continuous dividend yield (may be
    negative, like ``r``). ``option_type`` must be "call" or "put".
    """
    if np.any(np.asarray(S) <= 0):
        raise ValueError("S must be positive")
    if np.any(np.asarray(K) <= 0):
        raise ValueError("K must be positive")
    if np.any(np.asarray(T) < 0):
        raise ValueError("T cannot be negative")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma cannot be negative")
    for name, values in (("S", S), ("K", K), ("T", T), ("r", r), ("q", q), ("sigma", sigma)):
        _require_finite(name, values)
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError("option_type must be either 'call' or 'put'")

# --- Greek input validation ---


def validate_greek_inputs(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    q: ScalarOrArray,
    sigma: ScalarOrArray,
    option_type: OptionType,
) -> None:
    """Validate inputs for the Greek functions.

    Same checks as :func:`validate_pricing_inputs`, plus ``T > 0`` and
    ``sigma > 0`` because the Greek formulas divide by ``sqrt(T)`` and
    ``sigma``.
    """
    validate_pricing_inputs(S, K, T, r, q, sigma, option_type)
    if np.any(np.asarray(T) <= 0):
        raise ValueError("T must be positive when calculating Greeks")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive when calculating Greeks")
