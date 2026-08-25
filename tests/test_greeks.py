"""Tests for the Black-Scholes Greeks."""

import numpy as np
import pytest

from engine.core import (
    all_greeks,
    black_scholes_price,
    delta,
    gamma,
    rho,
    theta,
    vega,
)

# Reference values for S=100, K=100, T=1 year, r=5%, sigma=20% (Hull).
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20

REF = {
    "call": {
        "delta": 0.6368306511756191,
        "gamma": 0.018762017345846895,
        "vega": 37.52403469169379,
        "theta": -6.414027546438197,
        "rho": 53.232481545376345,
    },
    "put": {
        "delta": -0.3631693488243809,
        "gamma": 0.018762017345846895,
        "vega": 37.52403469169379,
        "theta": -1.657880423934626,
        "rho": -41.89046090469506,
    },
}

BASE = dict(S=S, K=K, T=T, r=R, sigma=SIGMA, option_type="call")


def _fd(function, x, h=1e-5):
    """Central finite difference of `function` at `x`."""
    return (function(x + h) - function(x - h)) / (2 * h)


# --- Hull reference values ---------------------------------------------------


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_hull_reference_greeks(greek, option_type):
    value = all_greeks(
        S=S, K=K, T=T, r=R, sigma=SIGMA, option_type=option_type
    )[greek]
    assert value == pytest.approx(REF[option_type][greek], rel=1e-9)


def test_all_greeks_returns_all_five():
    result = all_greeks(**BASE)
    assert set(result) == {"delta", "gamma", "vega", "theta", "rho"}


# --- Put-call parity relationships -------------------------------------------


def test_delta_put_call_parity():
    d_call = delta(**BASE)
    d_put = delta(**{**BASE, "option_type": "put"})
    assert d_call - d_put == pytest.approx(1.0, rel=1e-12)


def test_gamma_is_identical_for_call_and_put():
    g_call = gamma(**BASE)
    g_put = gamma(**{**BASE, "option_type": "put"})
    assert g_call == pytest.approx(g_put, rel=1e-12)


def test_vega_is_identical_for_call_and_put():
    v_call = vega(**BASE)
    v_put = vega(**{**BASE, "option_type": "put"})
    assert v_call == pytest.approx(v_put, rel=1e-12)


def test_theta_put_call_parity():
    # Differentiating put-call parity gives theta_c - theta_p = -r*K*exp(-rT).
    t_call = theta(**BASE)
    t_put = theta(**{**BASE, "option_type": "put"})
    assert t_call - t_put == pytest.approx(-R * K * np.exp(-R * T), rel=1e-12)


def test_rho_put_call_parity():
    # Differentiating put-call parity gives rho_c - rho_p = K*T*exp(-rT).
    r_call = rho(**BASE)
    r_put = rho(**{**BASE, "option_type": "put"})
    assert r_call - r_put == pytest.approx(K * T * np.exp(-R * T), rel=1e-12)


# --- Sign conventions --------------------------------------------------------


def test_call_delta_between_zero_and_one():
    assert 0.0 < delta(**BASE) < 1.0


def test_put_delta_between_minus_one_and_zero():
    params = dict(BASE, option_type="put")
    assert -1.0 < delta(**params) < 0.0


def test_gamma_and_vega_are_positive():
    assert gamma(**BASE) > 0.0
    assert vega(**BASE) > 0.0


def test_call_theta_is_negative_for_typical_params():
    assert theta(**BASE) < 0.0


def test_call_rho_positive_put_rho_negative():
    assert rho(**BASE) > 0.0
    assert rho(**dict(BASE, option_type="put")) < 0.0


def test_deep_itm_call_delta_tends_to_one():
    assert delta(**dict(BASE, S=1e6)) == pytest.approx(1.0, abs=1e-6)


def test_deep_otm_call_delta_tends_to_zero():
    assert delta(**dict(BASE, S=1e-6)) == pytest.approx(0.0, abs=1e-6)


# --- Validation --------------------------------------------------------------


def test_greeks_reject_zero_time():
    with pytest.raises(ValueError):
        gamma(**dict(BASE, T=0.0))


def test_greeks_reject_zero_volatility():
    with pytest.raises(ValueError):
        vega(**dict(BASE, sigma=0.0))


def test_greeks_reject_negative_time():
    with pytest.raises(ValueError):
        delta(**dict(BASE, T=-1.0))


def test_greeks_reject_non_finite_inputs():
    with pytest.raises(ValueError):
        delta(**dict(BASE, S=float("nan")))


# --- Cross-checks against finite differences ---------------------------------


@pytest.mark.parametrize(
    "greek_name,bump_param",
    [("delta", "S"), ("vega", "sigma"), ("rho", "r")],
)
def test_analytic_greek_matches_finite_difference(greek_name, bump_param):
    analytic = all_greeks(**BASE)[greek_name]
    numeric = _fd(
        lambda x: black_scholes_price(**{**BASE, bump_param: x}),
        BASE[bump_param],
    )
    assert numeric == pytest.approx(analytic, rel=1e-4)


def test_theta_matches_finite_difference():
    # Theta is -dV/dT, hence the minus sign.
    analytic = theta(**BASE)
    numeric = -_fd(lambda x: black_scholes_price(**{**BASE, "T": x}), T)
    assert numeric == pytest.approx(analytic, rel=1e-4)


def test_gamma_matches_second_difference():
    analytic = gamma(**BASE)
    h = 1e-4
    numeric = (
        black_scholes_price(**dict(BASE, S=S + h))
        - 2.0 * black_scholes_price(**BASE)
        + black_scholes_price(**dict(BASE, S=S - h))
    ) / h**2
    assert numeric == pytest.approx(analytic, rel=1e-4)


# --- Array support -----------------------------------------------------------


def test_array_greeks_match_scalar():
    spots = np.array([90.0, 100.0, 110.0])
    result = all_greeks(S=spots, K=K, T=T, r=R, sigma=SIGMA, option_type="call")
    for i, spot in enumerate(spots):
        scalar = all_greeks(
            S=float(spot), K=K, T=T, r=R, sigma=SIGMA, option_type="call"
        )
        for name in REF["call"]:
            assert result[name][i] == pytest.approx(scalar[name], rel=1e-12)


# --- Dividend yield (q) ------------------------------------------------------


Q = 0.03


def test_delta_parity_with_dividends():
    d_call = delta(**{**BASE, "q": Q})
    d_put = delta(**{**BASE, "q": Q, "option_type": "put"})
    assert d_call - d_put == pytest.approx(np.exp(-Q * T), rel=1e-12)


def test_theta_parity_with_dividends():
    # Differentiating C - P = S*exp(-qT) - K*exp(-rT) gives
    # theta_c - theta_p = q*S*exp(-qT) - r*K*exp(-rT).
    t_call = theta(**{**BASE, "q": Q})
    t_put = theta(**{**BASE, "q": Q, "option_type": "put"})
    expected = Q * S * np.exp(-Q * T) - R * K * np.exp(-R * T)
    assert t_call - t_put == pytest.approx(expected, rel=1e-12)


def test_greeks_match_finite_differences_with_dividends():
    base = {**BASE, "q": Q}
    for greek_name, bump_param in [("delta", "S"), ("vega", "sigma"), ("rho", "r")]:
        analytic = all_greeks(**base)[greek_name]
        numeric = _fd(
            lambda x: black_scholes_price(**{**base, bump_param: x}),
            base[bump_param],
        )
        assert numeric == pytest.approx(analytic, rel=1e-4)
    analytic_theta = theta(**base)
    numeric_theta = -_fd(lambda x: black_scholes_price(**{**base, "T": x}), T)
    assert numeric_theta == pytest.approx(analytic_theta, rel=1e-4)


def test_greeks_reject_non_finite_dividend_yield():
    with pytest.raises(ValueError):
        delta(**{**BASE, "q": float("nan")})
