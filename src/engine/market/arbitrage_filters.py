"""Static no-arbitrage checks over an option chain.

Real markets quote many options for the same underlying, and stale or
crossed quotes can break basic no-arbitrage relationships. These checks
flag such quotes so they can be removed before fitting a volatility
surface, where a single bad quote would poison the whole fit.

All checks compare **mid prices** and take a ``tolerance``: a small apparent
violation inside the bid-ask spread is not exploitable arbitrage and should
not be flagged, so pass a tolerance of roughly half the average spread when
working with real data.

The classic static conditions implemented here:

- **Lower bound**: an option is never worth less than its intrinsic value.
- **Vertical spread**: call mids fall as the strike rises; put mids rise.
- **Butterfly (convexity)**: the price curve is convex in strike, so no
  strike's mid sits above the linear interpolation of its neighbours.
- **Put-call parity**: C - P ~ S*exp(-qT) - K*exp(-rT) at each strike.

Calendar-spread checks (across expiries) are not implemented; they
need more than one chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.core import option_intrinsic_value
from engine.market.models import OptionChain, OptionQuote

# --- Violation report ---

@dataclass(frozen=True)
class Violation:
    """One detected arbitrage violation, for reporting or filtering.

    Parameters:
        rule: which check caught it ("lower_bound", "vertical_spread",
            "butterfly", "put_call_parity")
        message: human-readable description
        excess: how far past the tolerance the violation is
        quotes: the quotes involved (used to remove them from the chain)
    """

    rule: str
    message: str
    excess: float
    quotes: tuple[OptionQuote, ...] = ()

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


# --- Lower bound checks ---

def check_lower_bounds(
    chain: OptionChain,
    tolerance: float = 0.0,
) -> list[Violation]:
    """Flag quotes whose mid price is below the intrinsic value."""
    violations = []
    for quote in chain.quotes:
        intrinsic = option_intrinsic_value(
            S=chain.spot, K=quote.strike, option_type=quote.option_type
        )
        excess = intrinsic - quote.mid
        if excess > tolerance:
            violations.append(
                Violation(
                    rule="lower_bound",
                    message=(
                        f"{quote.option_type} K={quote.strike}: mid {quote.mid:.4f} "
                        f"below intrinsic {intrinsic:.4f}"
                    ),
                    excess=excess,
                    quotes=(quote,),
                )
            )
    return violations


# --- Vertical spread checks ---

def check_vertical_spread(
    chain: OptionChain,
    tolerance: float = 0.0,
) -> list[Violation]:
    """Flag strikes where call (put) mids do not fall (rise) with strike.

    Vertical-spread arbitrage requires C(K1) >= C(K2) for K1 < K2, and
    P(K1) <= P(K2) for K1 < K2.
    """
    violations = []
    for option_type in ("call", "put"):
        quotes = sorted(chain.of_type(option_type), key=lambda q: q.strike)
        for low, high in zip(quotes, quotes[1:]):
            if option_type == "call":
                # Calls must be cheaper at higher strikes.
                excess = high.mid - low.mid
                clause = (
                    f"C({low.strike})={low.mid:.4f} must be >= "
                    f"C({high.strike})={high.mid:.4f}"
                )
            else:
                # Puts must be more expensive at higher strikes.
                excess = low.mid - high.mid
                clause = (
                    f"P({low.strike})={low.mid:.4f} must be <= "
                    f"P({high.strike})={high.mid:.4f}"
                )
            if excess > tolerance:
                violations.append(
                    Violation(
                        rule="vertical_spread",
                        message=f"vertical-spread violation: {clause}",
                        excess=excess,
                        quotes=(low, high),
                    )
                )
    return violations


# --- Butterfly (convexity) checks ---

def check_butterfly(
    chain: OptionChain,
    tolerance: float = 0.0,
) -> list[Violation]:
    """Flag strike triples where mid prices are not convex in strike.

    Butterfly arbitrage requires the price curve to be convex: for any
    K1 < K2 < K3,

        price(K2) <= w * price(K1) + (1 - w) * price(K3)

    with w = (K3 - K2) / (K3 - K1). A violation means a negative
    butterfly: buying the wings and selling the body locks in risk-free
    profit.
    """
    violations = []
    for option_type in ("call", "put"):
        quotes = sorted(chain.of_type(option_type), key=lambda q: q.strike)
        for first, middle, last in zip(quotes, quotes[1:], quotes[2:]):
            k1, k2, k3 = first.strike, middle.strike, last.strike
            if k3 == k1:
                continue
            weight = (k3 - k2) / (k3 - k1)
            interpolated = weight * first.mid + (1.0 - weight) * last.mid
            excess = middle.mid - interpolated
            if excess > tolerance:
                violations.append(
                    Violation(
                        rule="butterfly",
                        message=(
                            f"{option_type} K={k2}: mid {middle.mid:.4f} above "
                            f"convex interpolation {interpolated:.4f} of "
                            f"K={k1} ({first.mid:.4f}) and K={k3} ({last.mid:.4f})"
                        ),
                        excess=excess,
                        quotes=(first, middle, last),
                    )
                )
    return violations


# --- Put-call parity checks ---

def check_put_call_parity(
    chain: OptionChain,
    r: float,
    q: float = 0.0,
    tolerance: float = 0.0,
) -> list[Violation]:
    """Flag strikes where call-put parity is violated.

    For European options, C - P = S*exp(-qT) - K*exp(-rT). Listed options
    are American so parity holds only approximately; use a tolerance
    (roughly half the average spread) in practice.
    """
    violations = []
    time_to_expiry = chain.time_to_expiry()
    for strike in chain.strikes():
        call = chain.at_strike(strike, "call")
        put = chain.at_strike(strike, "put")
        if call is None or put is None:
            continue
        parity = chain.spot * np.exp(-q * time_to_expiry) - strike * np.exp(-r * time_to_expiry)
        actual = call.mid - put.mid
        excess = abs(actual - parity)
        if excess > tolerance:
            violations.append(
                Violation(
                    rule="put_call_parity",
                    message=(
                        f"K={strike}: C - P = {actual:.4f}, put-call parity "
                        f"expects {parity:.4f}"
                    ),
                    excess=excess,
                    quotes=(call, put),
                )
            )
    return violations


# --- Run everything ---

def check_all(
    chain: OptionChain,
    r: float,
    q: float = 0.0,
    tolerance: float = 0.0,
) -> list[Violation]:
    """Run every no-arbitrage check and return all violations found."""
    violations = []
    violations += check_lower_bounds(chain, tolerance)
    violations += check_vertical_spread(chain, tolerance)
    violations += check_butterfly(chain, tolerance)
    violations += check_put_call_parity(chain, r, q, tolerance)
    return violations


def remove_arbitrage_violations(
    chain: OptionChain,
    r: float,
    q: float = 0.0,
    tolerance: float = 0.0,
) -> OptionChain:
    """Return a copy of the chain with quotes involved in violations removed.

    A quote that breaks any check is dropped, along with its counterpart in
    a parity violation. Quotes that are not involved are kept.
    """
    violations = check_all(chain, r, q, tolerance)
    to_remove = set()
    for violation in violations:
        to_remove.update(violation.quotes)

    return OptionChain(
        symbol=chain.symbol,
        as_of=chain.as_of,
        spot=chain.spot,
        expiry=chain.expiry,
        quotes=[q for q in chain.quotes if q not in to_remove],
    )
