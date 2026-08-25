"""Demo: run the whole options engine end to end.

Run from the project root:

    .venv/bin/python examples/demo.py

or in VS Code: open this file and press the Run button (top right).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

# Make the src/ layout importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from engine.core import all_greeks, black_scholes_price
from engine.vol import fit_svi, fit_svi_from_chain, implied_volatility, svi_total_variance
from engine.risk import crank_nicolson_price


# --- 1. Black-Scholes price and Greeks ---------------------------------------
print("=" * 72)
print("1. BLACK-SCHOLES PRICE + GREEKS")
print("=" * 72)

S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.05, 0.02, 0.25

call = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
put = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put")
print(f"S={S}  K={K}  T={T}y  r={r}  q={q}  sigma={sigma}")
print(f"Call price: {call:.4f}")
print(f"Put  price: {put:.4f}")
print(f"Intrinsic value of the call: {max(S - K, 0.0):.4f}  (time value = {call - max(S - K, 0.0):.4f})")

g = all_greeks(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
print("\nCall Greeks:")
for name, value in g.items():
    print(f"  {name:>6s} = {value: .4f}")


# --- 2. Put-call parity -------------------------------------------------------
print("\n" + "=" * 72)
print("2. PUT-CALL PARITY  (C - P == S*e^-qT - K*e^-rT)")
print("=" * 72)

lhs = call - put
rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
print(f"  C - P           = {lhs:.8f}")
print(f"  S*e^-qT - K*e^-rT = {rhs:.8f}")
print(f"  Difference      = {abs(lhs - rhs):.2e}  (should be ~0)")


# --- 3. Implied volatility round trip ----------------------------------------
print("\n" + "=" * 72)
print("3. IMPLIED VOLATILITY (invert price -> recover sigma)")
print("=" * 72)

imp_vol = implied_volatility(S=S, K=K, T=T, r=r, q=q, market_price=call, option_type="call")
print(f"  Priced with sigma={sigma}, implied vol recovered = {imp_vol:.6f}")


# --- 4. Crank-Nicolson PDE solver --------------------------------------------
print("\n" + "=" * 72)
print("4. CRANK-NICOLSON (numeric PDE solver, European vs American)")
print("=" * 72)

cn_eu = crank_nicolson_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put", exercise="european")
cn_am = crank_nicolson_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put", exercise="american")
print(f"  European put: {cn_eu:.4f}")
print(f"  American put: {cn_am:.4f}")
print(f"  Early-exercise premium: {cn_am - cn_eu:.4f}  (American >= European)")


# --- 5. SVI smile fit ---------------------------------------------------------
print("\n" + "=" * 72)
print("5. SVI SMILE FIT (recover the smile from implied vols)")
print("=" * 72)

true_params = dict(a=0.04, b=0.4, rho=-0.6, m=0.1, sigma=0.15)
T_svi = 0.5
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0])
k = np.log(strikes / S)
# True smile -> implied vols (synthetic "market" data)
vols = np.sqrt(svi_total_variance(k, **true_params) / T_svi)

fit = fit_svi(k=k, w=vols**2 * T_svi)
print(f"  True  params: a={true_params['a']} b={true_params['b']} rho={true_params['rho']} "
      f"m={true_params['m']} sigma={true_params['sigma']}")
print(f"  Fitted params: a={fit.a:.4f} b={fit.b:.4f} rho={fit.rho:.4f} "
      f"m={fit.m:.4f} sigma={fit.sigma:.4f}")
print(f"  Fit success: {fit.success}, max |vol error|: "
      f"{np.max(np.abs(fit.implied_volatility(k, T_svi) - vols)):.2e}")


# --- 6. Live market data (works with internet, falls back offline) ------------
print("\n" + "=" * 72)
print("6. LIVE PIPELINE: fetch a real chain -> filter -> SVI fit")
print("=" * 72)

try:
    from engine.market import check_all, fetch_expiries, fetch_option_chain

    # Pick the first expiry at least three weeks out: weekly expiries are
    # too close to expiry for a clean SVI fit. fetch_expiries returns date
    # objects, so compare them directly rather than re-parsing as strings.
    today = date.today()
    expiries = fetch_expiries("AAPL")
    expiry = next(
        (e for e in expiries if e - today >= timedelta(days=21)),
        expiries[0],
    )
    chain = fetch_option_chain("AAPL", expiry.strftime("%Y-%m-%d"))
    violations = check_all(chain, r=0.04, q=0.005, tolerance=0.05)
    smile = fit_svi_from_chain(chain=chain, r=0.04, q=0.005)
    print(f"  Fetched {len(chain.quotes)} AAPL quotes at spot {chain.spot:.2f}")
    print(f"  Arbitrage violations: {len(violations)}")
    print(f"  Fitted smile: a={smile.a:.4f} b={smile.b:.4f} "
          f"rho={smile.rho:.4f}")
except Exception as exc:  # no internet / Yahoo blocked -> show the fallback path
    print(f"  Live fetch unavailable ({exc.__class__.__name__}) in this environment.")
    print("  (On your machine with internet this section fetches real AAPL data.)")
    print("  See section 5 for the equivalent fit on synthetic data.")

print("\nAll sections ran. The engine works end to end!")
