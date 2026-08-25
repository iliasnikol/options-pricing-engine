"""Risk module: numerical pricing methods (finite-difference PDE solver).

Closed-form Black-Scholes cannot price American options, so this module
exists to solve the pricing PDE numerically instead, which handles early
exercise and can be extended to other path-dependent features later.
"""

from engine.risk.crank_nicolson import crank_nicolson_price

__all__ = ["crank_nicolson_price"]
