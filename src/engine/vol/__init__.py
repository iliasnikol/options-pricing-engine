"""Volatility module: implied volatility and SVI smile fitting.

Market prices quote in premium, not volatility, so this module bridges the
two: :func:`implied_volatility` recovers sigma from a single quote, and the
SVI fitter turns a whole chain of recovered vols into one smooth curve.
Import the public API from here instead of the individual modules.
"""

from engine.vol.implied_vol import implied_volatility
from engine.vol.svi import (
    SviFit,
    fit_svi,
    fit_svi_from_chain,
    svi_implied_volatility,
    svi_total_variance,
)

__all__ = [
    "implied_volatility",
    "SviFit",
    "fit_svi",
    "fit_svi_from_chain",
    "svi_implied_volatility",
    "svi_total_variance",
]
