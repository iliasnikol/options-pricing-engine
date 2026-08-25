"""SVI (Stochastic Volatility Inspired) volatility smile fitting.

The raw SVI parameterisation (Gatheral) models total implied variance as a
function of log-moneyness k = ln(K/S):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

The five parameters have clear meanings:

    a      level of the smile (ATM variance)
    b      steepness of the wings
    rho    skew: rho < 0 raises the put side and lowers the call side
    m      horizontal position of the smile minimum
    sigma  curvature at the minimum (small = sharp V, large = flat U)

The minimum of the curve sits at k = m - rho*sigma/sqrt(1-rho^2) with value
a + b*sigma*sqrt(1-rho^2), so the box constraints a, b, sigma >= 0 and
|rho| < 1 are enough to guarantee a valid (never negative) total variance.

:func:`fit_svi` finds the parameters by bounded least squares against
observed total variances; :func:`fit_svi_from_chain` is the convenience
wrapper that computes implied vols from an option chain first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from engine.core.validation import ScalarOrArray
from engine.market.models import OptionChain
from engine.vol.implied_vol import implied_volatility

# --- SVI curve ---

def svi_total_variance(
    k: ScalarOrArray,
    *,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> ScalarOrArray:
    """Total implied variance w(k) under the raw SVI parameterisation.

    Parameters:
        k: log-moneyness ln(K/S), scalar or array
        a, b, rho, m, sigma: the SVI parameters

    Raises:
        ValueError: if the parameters violate the validity constraints
            (a, b, sigma >= 0 and -1 < rho < 1).
    """
    if a < 0 or b < 0 or sigma < 0:
        raise ValueError("a, b and sigma must be non-negative")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be strictly between -1 and 1")

    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_implied_volatility(
    k: ScalarOrArray,
    *,
    T: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> ScalarOrArray:
    """Implied volatility from the SVI curve: sqrt(w(k) / T)."""
    if T <= 0:
        raise ValueError("T must be positive")
    total_variance = svi_total_variance(k, a=a, b=b, rho=rho, m=m, sigma=sigma)
    return np.sqrt(total_variance / T)


# --- Fit result ---

@dataclass(frozen=True)
class SviFit:
    """A fitted SVI curve plus diagnostics from the least-squares solver.

    Parameters:
        a, b, rho, m, sigma: the fitted SVI parameters
        success: whether the solver reported convergence
        cost: the final sum of squared residuals (0 is a perfect fit)
        n_observations: how many data points were fitted
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    success: bool
    cost: float
    n_observations: int

    @property
    def params(self) -> tuple[float, float, float, float, float]:
        """The fitted parameters as (a, b, rho, m, sigma)."""
        return (self.a, self.b, self.rho, self.m, self.sigma)

    def total_variance(self, k: ScalarOrArray) -> ScalarOrArray:
        """Evaluate the fitted curve at log-moneyness k."""
        return svi_total_variance(
            k, a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma
        )

    def implied_volatility(self, k: ScalarOrArray, T: float) -> ScalarOrArray:
        """Evaluate the fitted implied volatility at log-moneyness k."""
        return svi_implied_volatility(
            k, T=T, a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma
        )


# --- Fitting helpers ---

def _initial_guess(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Read the observed smile to build a sensible starting point.

    a starts at the lowest observed variance, m at the strike where it
    happens, b at the average slope of the smile, sigma at the strike
    spacing, and rho at zero (no skew).
    """
    index_min = int(np.argmin(w))
    m0 = float(k[index_min])
    a0 = max(float(w[index_min]), 0.0)

    k_range = float(np.ptp(k))
    w_range = float(np.ptp(w))
    b0 = max(w_range / k_range, 0.05) if k_range > 0 else 0.1
    sigma0 = max(float(np.mean(np.diff(np.sort(k)))), 0.05) if len(k) > 1 else 0.1

    return np.array([a0, b0, 0.0, m0, sigma0])


# --- Least-squares fit ---

def fit_svi(
    *,
    k: np.ndarray,
    w: np.ndarray,
    initial_guess: np.ndarray | None = None,
    max_nfev: int = 2000,
) -> SviFit:
    """Fit the raw SVI curve to observed total variances by least squares.

    Parameters:
        k: log-moneyness ln(K/S) of each observation
        w: total implied variance sigma_imp^2 * T of each observation
        initial_guess: optional starting parameters [a, b, rho, m, sigma]
        max_nfev: maximum number of solver function evaluations

    Returns:
        A fitted :class:`SviFit`. The solver nearly always reports success
        numerically, so check the curve against the data rather than relying
        on ``success`` alone.

    Raises:
        ValueError: if there are fewer than three observations or any
            k/w value is invalid.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if k.ndim != 1 or k.shape != w.shape:
        raise ValueError("k and w must be one-dimensional arrays of the same length")
    if len(k) < 3:
        raise ValueError("at least three observations are needed to fit SVI")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(w)):
        raise ValueError("k and w must contain only finite values")
    if np.any(w < 0):
        raise ValueError("total variance w cannot be negative")

    # Box constraints: a, b, sigma >= 0 and |rho| < 1 keep the curve valid.
    lower_bounds = np.array([0.0, 0.0, -0.99, -np.inf, 0.0])
    upper_bounds = np.array([np.inf, np.inf, 0.99, np.inf, np.inf])

    guess = _initial_guess(k, w) if initial_guess is None else np.asarray(initial_guess, dtype=float)
    if guess.shape != (5,):
        raise ValueError("initial_guess must have exactly five entries [a, b, rho, m, sigma]")

    def residuals(x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = x
        return svi_total_variance(k, a=a, b=b, rho=rho, m=m, sigma=sigma) - w

    result = least_squares(
        residuals,
        guess,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=max_nfev,
    )

    a, b, rho, m, sigma = result.x
    return SviFit(
        a=float(a),
        b=float(b),
        rho=float(rho),
        m=float(m),
        sigma=float(sigma),
        success=bool(result.success),
        cost=float(result.cost),
        n_observations=len(k),
    )


# --- Chain wrapper ---

def fit_svi_from_chain(
    *,
    chain: OptionChain,
    r: float,
    q: float = 0.0,
    max_nfev: int = 2000,
) -> SviFit:
    """Fit SVI to an option chain, computing implied vols from mid prices.

    Quotes with a non-positive mid price (missing bid/ask) and quotes whose
    implied volatility cannot be computed (e.g. below the no-arbitrage
    bound) are skipped.

    Parameters:
        chain: the option chain to fit (one expiry)
        r: continuously compounded risk-free rate
        q: continuous dividend yield (default 0)
        max_nfev: maximum number of solver function evaluations

    Returns:
        A fitted :class:`SviFit`.

    Raises:
        ValueError: if the chain has no positive time to expiry or fewer
            than three valid quotes remain after skipping.
    """
    time_to_expiry = chain.time_to_expiry()
    if time_to_expiry <= 0:
        raise ValueError("chain must have positive time to expiry")

    log_moneyness = []
    total_variance = []
    for quote in chain.quotes:
        if quote.mid <= 0:
            continue
        try:
            vol = implied_volatility(
                S=chain.spot,
                K=quote.strike,
                T=time_to_expiry,
                r=r,
                q=q,
                market_price=quote.mid,
                option_type=quote.option_type,
            )
        except ValueError:
            # Not priceable (e.g. below the no-arbitrage lower bound):
            # skip rather than poison the fit.
            continue
        log_moneyness.append(np.log(quote.strike / chain.spot))
        total_variance.append(vol**2 * time_to_expiry)

    if len(log_moneyness) < 3:
        raise ValueError("fewer than three valid quotes to fit SVI")

    return fit_svi(
        k=np.array(log_moneyness),
        w=np.array(total_variance),
        max_nfev=max_nfev,
    )
