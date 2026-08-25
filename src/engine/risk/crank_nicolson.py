"""Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

Unlike the closed-form Black-Scholes formula, a finite-difference solver can
price options whose value depends on the whole price path in space and time:
American options (early exercise), and later dividends or local volatility.

We work in log-space, x = ln(S), where the Black-Scholes PDE has constant
coefficients:

    dV/dtau = a * d2V/dx2 + b * dV/dx - r * V

with a = 0.5 * sigma^2 and b = r - q - 0.5 * sigma^2 (where q is the
continuous dividend yield), integrated forward in tau = T - t (time
remaining until expiry).

Crank-Nicolson is the theta-scheme with theta = 0.5: it averages the
explicit and implicit steps, making it second-order accurate in time and
space and unconditionally stable. The first few time steps use a fully
implicit (backward Euler) "Rannacher" start to damp the spurious oscillations
Crank-Nicolson produces near the payoff kink at the strike.

American options are handled by enforcing V >= intrinsic value at every time
step (early exercise is always available, so the option can never be worth
less than exercising it immediately).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from engine.core.validation import OptionType, validate_greek_inputs

# --- Scheme constants ---

#: Crank-Nicolson weights the explicit and implicit steps equally.
_THETA = 0.5

#: Rannacher smoothing: take this many fully implicit steps first.
_RANNACHER_STEPS = 3

#: Grid extends this many sigma*sqrt(T) either side of the spot price.
_GRID_HALF_WIDTH = 8.0


# --- Grid and payoff ---

def _boundary_values(
    spot_low: float,
    spot_high: float,
    tau: float,
    K: float,
    r: float,
    q: float,
    option_type: str,
) -> tuple[float, float]:
    """Dirichlet boundary values at time-to-maturity tau.

    Far below the strike a call is worthless and a put is worth the
    discounted strike; far above, the call is worth the discounted forward
    value of the stock (which pays dividends) and the put is worthless.
    """
    discounted_strike = K * np.exp(-r * tau)
    discounted_spot = spot_high * np.exp(-q * tau)

    if option_type == "call":
        return 0.0, discounted_spot - discounted_strike

    return discounted_strike - spot_low * np.exp(-q * tau), 0.0


def _build_system(
    alpha: float,
    gamma: float,
    diag_term: float,
    dt: float,
    theta: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the tridiagonal matrices of the theta-scheme.

    The time step solves:

        A V^{n+1} = B V^n

    where A is the implicit side and B the explicit side. A is returned in
    scipy's banded format (one sub- and one super-diagonal) for
    ``solve_banded``; B is a full matrix because it is used for a
    matrix-vector product. The boundary rows of A are identity (Dirichlet)
    and the interior rows do not couple to the boundary nodes; the known
    boundary contribution is added to the right-hand side by the caller.
    """
    a_banded = np.zeros((3, n_points))
    b_matrix = np.zeros((n_points, n_points))

    a_super = -theta * dt * gamma
    a_sub = -theta * dt * alpha
    a_diag = 1.0 + theta * dt * diag_term
    b_super = (1.0 - theta) * dt * gamma
    b_sub = (1.0 - theta) * dt * alpha
    b_diag = 1.0 - (1.0 - theta) * dt * diag_term

    idx = np.arange(n_points)
    for i in range(1, n_points - 1):
        a_banded[0, i] = a_super  # super-diagonal
        a_banded[1, i] = a_diag
        a_banded[2, i] = a_sub    # sub-diagonal
    b_matrix[idx[:-1], idx[1:]] = b_super
    b_matrix[idx, idx] = b_diag
    b_matrix[idx[1:], idx[:-1]] = b_sub

    # Dirichlet boundary rows: the boundary nodes are pinned to known values
    # and are not unknowns in the interior equations.
    # Set the diagonal of both boundary rows to identity.
    a_banded[1, 0] = 1.0
    a_banded[1, -1] = 1.0
    # Zero the off-diagonal entries of the boundary rows so that the solver
    # enforces v[0] = lower_b and v[-1] = upper_b exactly, without coupling
    # to the adjacent interior nodes.
    #
    # Scipy banded format: ab[0, j] = A[j-1, j] (super-diagonal at column j)
    #                      ab[2, j] = A[j+1, j] (sub-diagonal  at column j)
    #
    # The loop set a_banded[0, 1] = a_super  → A[0, 1] (boundary row 0,
    # coupling to the first interior node); zero it.
    a_banded[0, 1] = 0.0
    # The loop set a_banded[2, -2] = a_sub   → A[-1, -2] (boundary row n-1,
    # coupling to the last interior node); zero it.
    a_banded[2, -2] = 0.0

    return a_banded, b_matrix


# --- Crank-Nicolson price ---

def crank_nicolson_price(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float,
    option_type: OptionType,
    exercise: str = "european",
    n_steps: int = 400,
    m_steps: int = 200,
) -> float:
    """Price an option with the Crank-Nicolson finite-difference method.

    Parameters:
        S: current underlying price
        K: strike price
        T: time to expiry in years
        r: continuously compounded risk-free rate
        q: continuous dividend yield (default 0)
        sigma: annualised volatility
        option_type: either "call" or "put"
        exercise: either "european" (default) or "american"
        n_steps: number of space steps in the log-price grid
        m_steps: number of time steps to expiry

    Returns:
        The option price at the spot price S.

    Raises:
        ValueError: if the inputs are invalid or the grid is too coarse.
    """
    option_type = option_type.lower()
    exercise = exercise.lower()

    if exercise not in {"european", "american"}:
        raise ValueError("exercise must be either 'european' or 'american'")
    validate_greek_inputs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type)
    if n_steps < 10 or m_steps < 10:
        raise ValueError("n_steps and m_steps must each be at least 10")

    # Log-space grid centred on ln(S), wide enough that the analytic
    # boundary conditions are a good approximation.
    vol_time = sigma * np.sqrt(T)
    x_min = np.log(S) - _GRID_HALF_WIDTH * vol_time
    x_max = np.log(S) + _GRID_HALF_WIDTH * vol_time
    n_points = n_steps + 1
    dx = (x_max - x_min) / n_steps
    dt = T / m_steps
    grid = np.linspace(x_min, x_max, n_points)
    spot = np.exp(grid)

    # Payoff at expiry (tau = 0) and the intrinsic value used for the
    # American early-exercise constraint.
    if option_type == "call":
        value = np.maximum(spot - K, 0.0)
        intrinsic = lambda s: np.maximum(s - K, 0.0)  # noqa: E731
    else:
        value = np.maximum(K - spot, 0.0)
        intrinsic = lambda s: np.maximum(K - s, 0.0)  # noqa: E731

    # Constant coefficients of the log-space PDE. With a dividend yield the
    # drift becomes r - q instead of r.
    a = 0.5 * sigma**2
    b = r - q - 0.5 * sigma**2

    # Central-difference stencil coefficients.
    alpha = a / dx**2 - b / (2 * dx)  # coefficient of V[i-1]
    gamma = a / dx**2 + b / (2 * dx)  # coefficient of V[i+1]
    diag_term = 2.0 * a / dx**2 + r   # coefficient of V[i]

    # Matrices for the Rannacher (fully implicit) start and the
    # Crank-Nicolson steps that follow.
    a_euler, b_euler = _build_system(alpha, gamma, diag_term, dt, 1.0, n_points)
    a_cn, b_cn = _build_system(alpha, gamma, diag_term, dt, _THETA, n_points)

    # Integrate forward in tau from the payoff at tau = 0 to expiry T.
    for step in range(1, m_steps + 1):
        tau = step * dt
        lower_b, upper_b = _boundary_values(spot[0], spot[-1], tau, K, r, q, option_type)

        if step <= _RANNACHER_STEPS:
            a_matrix, b_matrix, theta = a_euler, b_euler, 1.0
        else:
            a_matrix, b_matrix, theta = a_cn, b_cn, _THETA

        rhs = b_matrix @ value
        # Move the known boundary values into the right-hand side.
        rhs[1] += theta * dt * alpha * lower_b
        rhs[-2] += theta * dt * gamma * upper_b
        rhs[0] = lower_b
        rhs[-1] = upper_b

        value = solve_banded((1, 1), a_matrix, rhs)

        # American options can be exercised early: the option can never be
        # worth less than its intrinsic value.
        if exercise == "american":
            value[1:-1] = np.maximum(value[1:-1], intrinsic(spot[1:-1]))

    # The spot price sits exactly on the grid midpoint.
    return float(value[n_points // 2])
