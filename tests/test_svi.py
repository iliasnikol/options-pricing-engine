"""Tests for the SVI curve and the least-squares fitter."""

from datetime import date

import numpy as np
import pytest

from engine.core import black_scholes_price
from engine.market import OptionChain, OptionQuote
from engine.vol import (
    fit_svi,
    fit_svi_from_chain,
    svi_implied_volatility,
    svi_total_variance,
)

# A hand-picked SVI curve with a realistic equity-style negative skew.
A, B, RHO, M, SIGMA = 0.04, 0.4, -0.6, 0.1, 0.15
PARAMS = dict(a=A, b=B, rho=RHO, m=M, sigma=SIGMA)


def make_chain(spot=100.0, strikes=None, vol_fn=None, r=0.05, q=0.0, T=1.0):
    """A zero-spread chain whose mids are priced with the given vol function."""
    if strikes is None:
        strikes = np.arange(80.0, 121.0, 5.0)
    if vol_fn is None:
        vol_fn = lambda k: np.sqrt(svi_total_variance(k, **PARAMS) / T)  # noqa: E731
    as_of, expiry = date(2025, 1, 16), date(2026, 1, 16)  # exactly one year
    quotes = []
    for strike in strikes:
        k = np.log(strike / spot)
        vol = vol_fn(k)
        for option_type in ("call", "put"):
            mid = black_scholes_price(
                S=spot, K=float(strike), T=T, r=r, q=q, sigma=vol, option_type=option_type
            )
            quotes.append(
                OptionQuote(
                    symbol="TEST", expiry=expiry, strike=float(strike),
                    option_type=option_type, bid=mid, ask=mid,
                )
            )
    return OptionChain(
        symbol="TEST", as_of=as_of, spot=spot, expiry=expiry, quotes=quotes,
    )


# --- SVI curve ---------------------------------------------------------------


def test_curve_is_never_negative():
    k = np.linspace(-3.0, 3.0, 601)
    w = svi_total_variance(k, **PARAMS)
    assert np.all(w >= 0)


def test_curve_minimum_matches_analytic_formula():
    # The minimum sits at k = m - rho*sigma/sqrt(1-rho^2) with value
    # a + b*sigma*sqrt(1-rho^2).
    k_min = M - RHO * SIGMA / np.sqrt(1 - RHO**2)
    w_min = A + B * SIGMA * np.sqrt(1 - RHO**2)

    k = np.linspace(-1.0, 1.0, 2001)
    w = svi_total_variance(k, **PARAMS)
    assert k[np.argmin(w)] == pytest.approx(k_min, abs=1e-3)
    # The grid rarely lands exactly on the minimum, so allow a small gap.
    assert w.min() == pytest.approx(w_min, abs=1e-5)


def test_curve_is_symmetric_when_rho_is_zero():
    symmetric = dict(a=A, b=B, rho=0.0, m=0.0, sigma=SIGMA)
    k = np.linspace(-1.0, 1.0, 101)
    w = svi_total_variance(k, **symmetric)
    assert np.allclose(w, svi_total_variance(-k, **symmetric))


def test_negative_rho_raises_left_wing():
    # rho < 0 lifts the put side (k < 0) above the call side (k > 0).
    w = svi_total_variance(np.array([-0.2, 0.2]), **PARAMS)
    assert w[0] > w[1]


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        svi_total_variance(0.0, a=-0.1, b=B, rho=RHO, m=M, sigma=SIGMA)
    with pytest.raises(ValueError):
        svi_total_variance(0.0, a=A, b=-0.1, rho=RHO, m=M, sigma=SIGMA)
    with pytest.raises(ValueError):
        svi_total_variance(0.0, a=A, b=B, rho=1.0, m=M, sigma=SIGMA)
    with pytest.raises(ValueError):
        svi_total_variance(0.0, a=A, b=B, rho=RHO, m=M, sigma=-0.1)


def test_implied_volatility_round_trip():
    k = np.linspace(-0.5, 0.5, 11)
    w = svi_total_variance(k, **PARAMS)
    vol = svi_implied_volatility(k, T=1.0, **PARAMS)
    assert np.allclose(vol**2, w, rtol=1e-12)


def test_implied_volatility_requires_positive_time():
    with pytest.raises(ValueError):
        svi_implied_volatility(0.0, T=0.0, **PARAMS)


# --- Least-squares fit -------------------------------------------------------


def test_fit_recovers_parameters_from_clean_data():
    k = np.linspace(-0.5, 0.5, 41)
    w = svi_total_variance(k, **PARAMS)
    fit = fit_svi(k=k, w=w)

    assert fit.success is True
    assert fit.n_observations == len(k)
    # The curve itself is recovered to high precision.
    assert np.max(np.abs(fit.total_variance(k) - w) / w) < 1e-4
    # And the parameters land back on the true values.
    assert fit.a == pytest.approx(A, abs=1e-3)
    assert fit.b == pytest.approx(B, abs=1e-3)
    assert fit.rho == pytest.approx(RHO, abs=1e-3)
    assert fit.m == pytest.approx(M, abs=1e-3)
    assert fit.sigma == pytest.approx(SIGMA, abs=1e-3)


def test_fit_is_robust_to_small_noise():
    k = np.linspace(-0.5, 0.5, 41)
    w = svi_total_variance(k, **PARAMS)
    rng = np.random.default_rng(42)
    w_noisy = w + rng.normal(0.0, 1e-4, len(w))
    fit = fit_svi(k=k, w=w_noisy)
    assert np.max(np.abs(fit.total_variance(k) - w) / w) < 1e-2


def test_fit_accepts_custom_initial_guess():
    k = np.linspace(-0.5, 0.5, 21)
    w = svi_total_variance(k, **PARAMS)
    guess = np.array([0.1, 0.1, 0.0, 0.0, 0.1])
    fit = fit_svi(k=k, w=w, initial_guess=guess)
    assert np.max(np.abs(fit.total_variance(k) - w) / w) < 1e-3


def test_fit_rejects_bad_data():
    k = np.array([-0.2, 0.0, 0.2])
    with pytest.raises(ValueError):
        fit_svi(k=k[:2], w=k[:2])  # too few points
    with pytest.raises(ValueError):
        fit_svi(k=k, w=np.array([0.04, -0.01, 0.06]))  # negative variance
    with pytest.raises(ValueError):
        fit_svi(k=k, w=np.array([0.04, float("nan"), 0.06]))  # non-finite
    with pytest.raises(ValueError):
        fit_svi(k=k, w=np.array([0.04, 0.05, 0.06]), initial_guess=np.zeros(3))


# --- Chain wrapper -----------------------------------------------------------


def test_fit_from_chain_recovers_generated_smile():
    chain = make_chain()
    fit = fit_svi_from_chain(chain=chain, r=0.05)

    for quote in chain.quotes:
        k = np.log(quote.strike / chain.spot)
        expected = np.sqrt(svi_total_variance(k, **PARAMS))
        assert fit.implied_volatility(k, T=1.0) == pytest.approx(expected, rel=1e-3)


def test_fit_from_chain_approximates_linear_skew():
    # A linear-in-moneyness vol is not exactly SVI, but the fit should track
    # it closely across the chain.
    vol_fn = lambda k: 0.2 - 0.15 * k  # noqa: E731
    chain = make_chain(vol_fn=vol_fn)
    fit = fit_svi_from_chain(chain=chain, r=0.05)

    for quote in chain.quotes:
        k = np.log(quote.strike / chain.spot)
        assert fit.implied_volatility(k, T=1.0) == pytest.approx(vol_fn(k), abs=2e-3)


def test_fit_from_chain_skips_unpriceable_quotes():
    chain = make_chain()
    # A quote below its intrinsic value cannot be assigned an implied vol;
    # the wrapper must skip it rather than fail.
    bad = OptionQuote(
        symbol="TEST", expiry=date(2026, 1, 16), strike=95.0,
        option_type="call", bid=0.5, ask=0.5,
    )
    chain.quotes.append(bad)
    fit = fit_svi_from_chain(chain=chain, r=0.05)
    assert fit.success is True


def test_fit_from_chain_requires_positive_time():
    chain = make_chain()
    chain.as_of = chain.expiry  # zero time to expiry
    with pytest.raises(ValueError):
        fit_svi_from_chain(chain=chain, r=0.05)


def test_fit_from_chain_requires_enough_quotes():
    # One strike gives only two quotes (call + put), which is not enough.
    chain = make_chain(strikes=np.array([95.0]))
    with pytest.raises(ValueError):
        fit_svi_from_chain(chain=chain, r=0.05)


# --- Dividend yield (q) ------------------------------------------------------


def test_fit_from_chain_with_dividends():
    q = 0.03
    chain = make_chain(r=0.05, q=q)
    fit = fit_svi_from_chain(chain=chain, r=0.05, q=q)
    for quote in chain.quotes:
        k = np.log(quote.strike / chain.spot)
        expected = np.sqrt(svi_total_variance(k, **PARAMS))
        assert fit.implied_volatility(k, T=1.0) == pytest.approx(expected, rel=1e-3)


def test_fit_from_chain_requires_correct_dividend_yield():
    # Getting q wrong distorts every implied vol and visibly corrupts the
    # fitted smile, even though the fit itself still "succeeds".
    q = 0.08
    chain = make_chain(r=0.05, q=q)
    fit_right = fit_svi_from_chain(chain=chain, r=0.05, q=q)
    fit_wrong = fit_svi_from_chain(chain=chain, r=0.05, q=0.0)

    max_error = max(
        abs(
            fit_wrong.implied_volatility(np.log(quote.strike / chain.spot), T=1.0)
            - np.sqrt(svi_total_variance(np.log(quote.strike / chain.spot), **PARAMS))
        )
        for quote in chain.quotes
    )
    assert max_error > 1e-2
