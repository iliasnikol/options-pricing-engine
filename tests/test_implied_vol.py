"""Tests for implied volatility inversion."""

import numpy as np
import pytest

from engine.core import black_scholes_price, european_lower_bound
from engine.vol import implied_volatility

S, K, T, R = 100.0, 100.0, 1.0, 0.05


def ivol(**overrides):
    params = dict(S=S, K=K, T=T, r=R, market_price=10.0, option_type="call")
    params.update(overrides)
    return implied_volatility(**params)


# --- Round-trip: price with sigma, then invert back to sigma ----------------


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_recovers_volatility(sigma, option_type):
    price = black_scholes_price(
        S=S, K=K, T=T, r=R, sigma=sigma, option_type=option_type
    )
    recovered = implied_volatility(
        S=S, K=K, T=T, r=R, market_price=price, option_type=option_type
    )
    assert recovered == pytest.approx(sigma, rel=1e-6)


@pytest.mark.parametrize(
    "spot,time,sigma",
    [(90.0, 0.25, 0.3), (110.0, 2.0, 0.15), (80.0, 0.5, 0.6), (120.0, 0.1, 0.35)],
)
def test_round_trip_off_the_money(spot, time, sigma):
    price = black_scholes_price(
        S=spot, K=K, T=time, r=R, sigma=sigma, option_type="call"
    )
    recovered = implied_volatility(
        S=spot, K=K, T=time, r=R, market_price=price, option_type="call"
    )
    assert recovered == pytest.approx(sigma, rel=1e-6)


def test_round_trip_deep_itm_put():
    sigma = 0.3
    price = black_scholes_price(
        S=50.0, K=K, T=T, r=R, sigma=sigma, option_type="put"
    )
    recovered = implied_volatility(
        S=50.0, K=K, T=T, r=R, market_price=price, option_type="put"
    )
    assert recovered == pytest.approx(sigma, rel=1e-6)


def test_round_trip_off_the_money_put():
    sigma = 0.4
    price = black_scholes_price(
        S=150.0, K=K, T=0.5, r=R, sigma=sigma, option_type="put"
    )
    recovered = implied_volatility(
        S=150.0, K=K, T=0.5, r=R, market_price=price, option_type="put"
    )
    assert recovered == pytest.approx(sigma, rel=1e-6)


def test_custom_initial_guess():
    sigma = 0.2
    price = black_scholes_price(
        S=S, K=K, T=T, r=R, sigma=sigma, option_type="call"
    )
    recovered = implied_volatility(
        S=S, K=K, T=T, r=R, market_price=price, option_type="call",
        initial_guess=0.35,
    )
    assert recovered == pytest.approx(sigma, rel=1e-6)


# --- Root-finder fallback ----------------------------------------------------


def test_brent_fallback_when_newton_fails():
    # A far-off initial guess lands where vega ~ 0, so Newton's derivative
    # guard trips and the Brent fallback must take over.
    sigma = 0.2
    price = black_scholes_price(
        S=S, K=K, T=T, r=R, sigma=sigma, option_type="call"
    )
    recovered = implied_volatility(
        S=S, K=K, T=T, r=R, market_price=price, option_type="call",
        initial_guess=45.0,
    )
    assert recovered == pytest.approx(sigma, rel=1e-6)


def test_implied_volatility_increases_with_price():
    vols = [
        implied_volatility(S=S, K=K, T=T, r=R, market_price=p, option_type="call")
        for p in (6.0, 10.450583572185565, 15.0)
    ]
    assert vols[0] < vols[1] < vols[2]


# --- No-arbitrage boundary behaviour -----------------------------------------


def test_price_at_lower_bound_gives_zero_volatility():
    lower = european_lower_bound(S=90.0, K=K, T=T, r=R, option_type="call")
    assert implied_volatility(
        S=90.0, K=K, T=T, r=R, market_price=lower, option_type="call"
    ) == pytest.approx(0.0)


def test_price_below_lower_bound_raises():
    # S=90 put is ITM: lower bound = K*exp(-rT) - S ~ 5.12, so a price of
    # 0.01 is below it and no volatility can reproduce it.
    with pytest.raises(ValueError, match="lower bound"):
        ivol(S=90.0, option_type="put", market_price=0.01)


def test_call_price_at_upper_bound_raises():
    with pytest.raises(ValueError, match="upper bound"):
        ivol(market_price=S)


def test_put_price_at_upper_bound_raises():
    with pytest.raises(ValueError, match="upper bound"):
        ivol(option_type="put", market_price=K * np.exp(-R * T))


# --- Input validation --------------------------------------------------------


def test_negative_price_raises():
    with pytest.raises(ValueError, match="negative"):
        ivol(market_price=-1.0)


def test_non_finite_price_raises():
    with pytest.raises(ValueError, match="finite"):
        ivol(market_price=float("nan"))


def test_zero_time_raises():
    with pytest.raises(ValueError, match="positive"):
        ivol(T=0.0)


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        ivol(option_type="straddle")


def test_bad_initial_guess_raises():
    with pytest.raises(ValueError, match="initial_guess"):
        ivol(initial_guess=-0.5)


# --- Dividend yield (q) ------------------------------------------------------


def test_round_trip_with_dividends():
    q = 0.03
    for option_type in ("call", "put"):
        price = black_scholes_price(
            S=S, K=K, T=T, r=R, q=q, sigma=0.2, option_type=option_type
        )
        recovered = implied_volatility(
            S=S, K=K, T=T, r=R, q=q, market_price=price, option_type=option_type
        )
        assert recovered == pytest.approx(0.2, rel=1e-6)


def test_price_at_dividend_adjusted_lower_bound_gives_zero_vol():
    q = 0.08
    # S=110 with q=8%: discounted spot ~101.5 > discounted strike ~95.1, so
    # the call's lower bound is positive and a price at it implies zero vol.
    lower = european_lower_bound(S=110.0, K=K, T=T, r=R, q=q, option_type="call")
    assert implied_volatility(
        S=110.0, K=K, T=T, r=R, q=q, market_price=lower, option_type="call"
    ) == pytest.approx(0.0)
