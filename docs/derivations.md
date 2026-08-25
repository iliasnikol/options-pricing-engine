# Mathematical Derivations

This document covers the mathematics behind every formula in the engine. It moves from stock price dynamics through the Black-Scholes PDE and its closed-form solution, to the Greek sensitivities and the log-space PDE the Crank-Nicolson solver uses.

---

## Contents

1. [Stock Price Dynamics](#1-stock-price-dynamics)
2. [Itô's Lemma](#2-itôs-lemma)
3. [Delta-Hedging and the Black-Scholes PDE](#3-delta-hedging-and-the-black-scholes-pde)
4. [Solving the PDE](#4-solving-the-pde)
5. [The Black-Scholes Closed-Form Prices](#5-the-black-scholes-closed-form-prices)
6. [Risk-Neutral Valuation](#6-risk-neutral-valuation)
7. [Continuous Dividend Yield](#7-continuous-dividend-yield)
8. [The Greeks](#8-the-greeks)
   - [Delta](#81-delta)
   - [Gamma](#82-gamma)
   - [Vega](#83-vega)
   - [Theta](#84-theta)
   - [Rho](#85-rho)
9. [The Log-Space PDE for the Crank-Nicolson Solver](#9-the-log-space-pde)
10. [Put-Call Parity](#10-put-call-parity)
11. [No-Arbitrage Bounds](#11-no-arbitrage-bounds)
12. [Implied Volatility](#12-implied-volatility)
13. [The SVI Parameterisation](#13-the-svi-parameterisation)

---

## 1. Stock Price Dynamics

The underlying stock price $S_t$ is modelled as a **geometric Brownian motion** (GBM):

$$dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mu$ | drift (expected return; not needed for pricing) |
| $\sigma > 0$ | annualised volatility |
| $W_t$ | a standard Brownian motion on a filtered probability space $(\Omega, \mathcal{F}, \mathbb{P})$ |

**Why GBM?** Because $dS_t \propto S_t$, it is returns rather than price changes that are normally distributed. The model can't produce negative prices. The increments $dW_t$ are independent with $dW_t \sim \mathcal{N}(0, dt)$.

Applying Itô's lemma to $\ln S_t$ gives the explicit solution:

$$S_T = S_0 \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)T + \sigma W_T\right]$$

so $\ln(S_T/S_0) \sim \mathcal{N}\!\left(\left(\mu - \frac{1}{2}\sigma^2\right)T,\ \sigma^2 T\right)$.

---

## 2. Itô's Lemma

Let $V(S, t)$ be a smooth function of the stock price and time. Itô's lemma extends the ordinary chain rule to stochastic processes:

$$dV = \frac{\partial V}{\partial t}\,dt
      + \frac{\partial V}{\partial S}\,dS
      + \frac{1}{2}\frac{\partial^2 V}{\partial S^2}\,(dS)^2$$

The key rule of stochastic calculus is $(dW_t)^2 = dt$ in the $L^2$ sense. This means $(dS)^2 = \sigma^2 S^2 \, dt$, since the $dt \cdot dW$ and $(dt)^2$ terms vanish. Substituting the GBM dynamics:

$$\boxed{dV = \left(\frac{\partial V}{\partial t}
              + \mu S \frac{\partial V}{\partial S}
              + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right)dt
           + \sigma S \frac{\partial V}{\partial S}\,dW_t}$$

The $dW_t$ term carries all the randomness in the option price. Delta-hedging removes it.

---

## 3. Delta-Hedging and the Black-Scholes PDE

### 3.1 The Hedged Portfolio

Construct a **self-financing portfolio** $\Pi$ that is long one option and short $\Delta$ units of the stock:

$$\Pi = V - \Delta \cdot S$$

Its instantaneous P&L is:

$$d\Pi = dV - \Delta \, dS$$

Substituting Itô's lemma and the GBM:

$$d\Pi = \left(\frac{\partial V}{\partial t}
              + \mu S \frac{\partial V}{\partial S}
              + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
              - \Delta \mu S\right)dt
       + \sigma S\!\left(\frac{\partial V}{\partial S} - \Delta\right)dW_t$$

### 3.2 Eliminating the Stochastic Term

Set:

$$\Delta = \frac{\partial V}{\partial S}$$

The $dW_t$ term vanishes. The portfolio is now instantaneously **risk-free**, regardless of $\mu$ or the realisation of the Brownian motion.

### 3.3 No-Arbitrage Condition

A risk-free portfolio must earn exactly the risk-free rate $r$, or there is an arbitrage:

$$d\Pi = r \,\Pi \, dt = r\!\left(V - \frac{\partial V}{\partial S} S\right)dt$$

### 3.4 The Black-Scholes PDE

Equating the $dt$ coefficients and simplifying:

$$\boxed{\frac{\partial V}{\partial t}
  + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
  + r S \frac{\partial V}{\partial S}
  - r V = 0}$$

This PDE holds for any European derivative whose value depends only on $S$ and $t$. The drift $\mu$ has dropped out entirely. Option prices are independent of the investor's expected return.

**Terminal and boundary conditions for a European call:**

$$V(S, T) = \max(S - K,\, 0), \quad
  V(0, t) = 0, \quad
  V(S, t) \sim S - K e^{-r(T-t)} \text{ as } S \to \infty$$

For a European put, replace the terminal condition with $\max(K - S, 0)$.

---

## 4. Solving the PDE

The Black-Scholes PDE can be reduced to the classical **heat equation** by a change of variables. Introduce:

$$x = \ln\frac{S}{K}, \quad
  \tau = \frac{1}{2}\sigma^2(T - t), \quad
  V(S, t) = K \, e^{\alpha x + \beta \tau} \, u(x, \tau)$$

with:

$$\alpha = -\frac{k - 1}{2}, \quad
  \beta  = -\frac{(k+1)^2}{4}, \quad
  k = \frac{2r}{\sigma^2}$$

After substitution, the PDE becomes:

$$\frac{\partial u}{\partial \tau} = \frac{\partial^2 u}{\partial x^2}$$

The terminal payoff $V(S, T) = \max(S - K, 0)$ translates into the initial condition $u(x, 0) = \max\!\left(e^{\frac{k+1}{2}x} - e^{\frac{k-1}{2}x}, 0\right)$.

The heat equation with this initial condition has a known Gaussian integral solution. Evaluating that integral and reversing the change of variables yields the closed-form formula in section 5. The derivation of the Gaussian integral itself is not reproduced here because the engine does not implement it — the code uses the final formula directly. What this change of variables *does* motivate directly is the log-space PDE in section 9, which the Crank-Nicolson solver discretises.

---

## 5. The Black-Scholes Closed-Form Prices

Define the standardised log-moneyness terms:

$$d_1 = \frac{\ln(S/K) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}$$

where $T$ is time to expiry and $N(\cdot)$ is the standard normal CDF.

**European call:**

$$\boxed{C = S\,N(d_1) - K e^{-rT} N(d_2)}$$

**European put:**

$$\boxed{P = K e^{-rT} N(-d_2) - S\,N(-d_1)}$$

### Interpretation of $d_1$ and $d_2$

- $N(d_2)$ is the risk-neutral probability that the call expires in-the-money: $\mathbb{Q}[S_T > K]$.
- $N(d_1)$ is the delta of the call, the hedge ratio needed to replicate the option.
- $K e^{-rT} N(d_2)$ is the present value of the strike payment, weighted by the probability it is made.
- $S\,N(d_1)$ is the present value of receiving the stock conditional on exercise.

---

## 6. Risk-Neutral Valuation

There is a second route to the same formula that avoids the PDE. By Girsanov's theorem there exists a probability measure $\mathbb{Q}$, equivalent to $\mathbb{P}$, under which discounted asset prices are martingales:

$$dS_t = r \, S_t \, dt + \sigma \, S_t \, d\tilde{W}_t$$

where $\tilde{W}_t = W_t + \frac{\mu - r}{\sigma} t$ is a $\mathbb{Q}$-Brownian motion. Under $\mathbb{Q}$, the stock drifts at the risk-free rate $r$, and the unique no-arbitrage price of any European derivative is:

$$V(S, t) = e^{-r(T-t)}\,\mathbb{E}^{\mathbb{Q}}\!\left[h(S_T) \mid \mathcal{F}_t\right]$$

For a call with $h(S_T) = \max(S_T - K, 0)$:

$$C = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[(S_T - K)^+\right]$$

Under $\mathbb{Q}$, $\ln S_T \sim \mathcal{N}\!\left(\ln S + (r - \frac{1}{2}\sigma^2)T,\ \sigma^2 T\right)$, so the expectation evaluates to $C = SN(d_1) - Ke^{-rT}N(d_2)$, the same formula as before.

---

## 7. Continuous Dividend Yield

With a continuous dividend yield $q$, the stock pays dividends at rate $q \, S \, dt$. This reduces the stock price and changes the self-financing condition. The risk-neutral drift becomes $r - q$:

$$dS_t = (r - q)\, S_t \, dt + \sigma \, S_t \, d\tilde{W}_t$$

The effect is equivalent to replacing $S$ with the dividend-adjusted spot $S e^{-qT}$ in the pricing formulas. The modified $d_1$ and $d_2$ are:

$$d_1 = \frac{\ln(S/K) + \left(r - q + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}$$

**European call with dividends:**

$$\boxed{C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)}$$

**European put with dividends:**

$$\boxed{P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)}$$

Setting $q = 0$ recovers the standard formulas. Every formula and Greek in this engine carries the $q$ parameter.

---

## 8. The Greeks

The Greeks measure how much the option price changes when one input moves. Let $n(\cdot) = N'(\cdot) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ be the standard normal density. The dividend-adjusted formulas are used throughout; set $q = 0$ for the no-dividend case.

Two identities appear repeatedly. Differentiating $d_1$ and $d_2$ with respect to $S$:

$$\frac{\partial d_1}{\partial S} = \frac{\partial d_2}{\partial S} = \frac{1}{S \sigma \sqrt{T}}$$

And a second: $S e^{-qT} n(d_1) = K e^{-rT} n(d_2)$. To see this, expand $n(d_2)$ using $d_2 = d_1 - \sigma\sqrt{T}$:

$$n(d_2) = n(d_1) \exp\!\left(d_1 \sigma\sqrt{T} - \tfrac{1}{2}\sigma^2 T\right)$$

Substituting $d_1 \sigma\sqrt{T} = \ln(S/K) + (r-q+\frac{1}{2}\sigma^2)T$ confirms the identity. It cancels the $n$-derivative terms in every Greek derivation below, keeping the algebra clean.

---

### 8.1 Delta

Delta is the change in option price per unit move in the underlying $S$.

**Call:**

$$\Delta_C = \frac{\partial C}{\partial S}
= e^{-qT} N(d_1) + S e^{-qT} n(d_1) \frac{\partial d_1}{\partial S}
  - K e^{-rT} n(d_2) \frac{\partial d_2}{\partial S}$$

Since $\partial d_1/\partial S = \partial d_2/\partial S = 1/(S\sigma\sqrt{T})$ and $S e^{-qT} n(d_1) = K e^{-rT} n(d_2)$, the last two terms cancel:

$$\boxed{\Delta_C = e^{-qT} N(d_1)}$$

**Put:**

$$\Delta_P = \frac{\partial P}{\partial S}
= -e^{-qT} N(-d_1) + K e^{-rT} n(-d_2) \frac{\partial d_2}{\partial S}
  - S e^{-qT} n(-d_1) \frac{\partial d_1}{\partial S}$$

By the same cancellation:

$$\boxed{\Delta_P = e^{-qT}\!\left(N(d_1) - 1\right) = -e^{-qT} N(-d_1)}$$

Note: $\Delta_C - \Delta_P = e^{-qT}$, which is the derivative of put-call parity with respect to $S$.

---

### 8.2 Gamma

Gamma is the second derivative of the option price with respect to $S$. It measures how fast Delta changes, and so how often the hedge needs rebalancing.

Gamma is the same for calls and puts. Differentiating the parity relation $C - P = S e^{-qT} - K e^{-rT}$ twice with respect to $S$ gives zero, confirming this. Taking the derivative of $\Delta_C$:

$$\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{\partial \Delta_C}{\partial S}
= e^{-qT} n(d_1) \frac{\partial d_1}{\partial S}$$

$$\boxed{\Gamma = \frac{e^{-qT} n(d_1)}{S \sigma \sqrt{T}}}$$

Gamma is always positive and peaks at-the-money, where a small move in $S$ changes Delta the most and so makes the hedge most expensive to rebalance.

---

### 8.3 Vega

Vega is the sensitivity of the option price to a change in volatility $\sigma$. It is the same for calls and puts.

$$\mathcal{V} = \frac{\partial C}{\partial \sigma}
= S e^{-qT} n(d_1) \frac{\partial d_1}{\partial \sigma}
  - K e^{-rT} n(d_2) \frac{\partial d_2}{\partial \sigma}$$

Since $\partial d_2/\partial \sigma = \partial d_1/\partial \sigma - \sqrt{T}$:

$$\frac{\partial C}{\partial \sigma}
= S e^{-qT} n(d_1) \frac{\partial d_1}{\partial \sigma}
  - K e^{-rT} n(d_2)\!\left(\frac{\partial d_1}{\partial \sigma} - \sqrt{T}\right)$$

The identity $S e^{-qT} n(d_1) = K e^{-rT} n(d_2)$ cancels the $\partial d_1/\partial \sigma$ terms:

$$\boxed{\mathcal{V} = S e^{-qT} \sqrt{T}\, n(d_1)}$$

The engine returns Vega per unit change in $\sigma$. Moving volatility from 20% to 21% changes the price by roughly $\mathcal{V}/100$.

---

### 8.4 Theta

Theta is the rate of change of the option price as time passes. It is reported as $\partial V/\partial t$, which is negative because the option loses value as expiry approaches (equivalently, $-\partial V/\partial T$).

**Call:**

$$\Theta_C = \frac{\partial C}{\partial t}
= \frac{\partial}{\partial t}\!\left[S e^{-qT} N(d_1) - K e^{-rT} N(d_2)\right]$$

Because $T$ appears inside $d_1$, $d_2$, and both discount factors there are several terms. The identity $S e^{-qT} n(d_1) = K e^{-rT} n(d_2)$ cancels the $d$-derivative contributions, leaving:

$$\boxed{\Theta_C = -\frac{S e^{-qT} n(d_1)\,\sigma}{2\sqrt{T}}
  - r K e^{-rT} N(d_2)
  + q S e^{-qT} N(d_1)}$$

**Put:**

$$\boxed{\Theta_P = -\frac{S e^{-qT} n(d_1)\,\sigma}{2\sqrt{T}}
  + r K e^{-rT} N(-d_2)
  - q S e^{-qT} N(-d_1)}$$

The first term (the diffusion contribution) is identical for calls and puts and is always negative. The remaining terms flip sign between the two, reflecting the different cost-of-carry effects.

The engine returns Theta as an annual quantity. Divide by 365 to get daily theta.

---

### 8.5 Rho

Rho is the sensitivity of the option price to the risk-free rate $r$. A higher rate reduces the present value of the strike payment, which benefits calls and hurts puts.

**Call:**

$$\rho_C = \frac{\partial C}{\partial r}
= S e^{-qT} n(d_1) \frac{\partial d_1}{\partial r}
  + K T e^{-rT} N(d_2)
  - K e^{-rT} n(d_2) \frac{\partial d_2}{\partial r}$$

Since $\partial d_1/\partial r = \partial d_2/\partial r = \sqrt{T}/\sigma$ and $S e^{-qT} n(d_1) = K e^{-rT} n(d_2)$, the first and third terms cancel:

$$\boxed{\rho_C = K T e^{-rT} N(d_2)}$$

**Put:**

$$\boxed{\rho_P = -K T e^{-rT} N(-d_2)}$$

The engine returns Rho per unit change in $r$. Moving from 5% to 6% changes the price by roughly $\rho/100$.

---

## 9. The Log-Space PDE

The Crank-Nicolson solver works in log-price space $x = \ln S$ rather than price space. This gives constant PDE coefficients, which means a single tridiagonal system can be built once and reused at every time step rather than rebuilt from scratch.

### 9.1 Transformation

Under $x = \ln S$ (so $S = e^x$):

$$\frac{\partial V}{\partial S} = \frac{\partial V}{\partial x} \cdot \frac{1}{S}, \quad
  \frac{\partial^2 V}{\partial S^2} = \frac{1}{S^2}\!\left(\frac{\partial^2 V}{\partial x^2}
  - \frac{\partial V}{\partial x}\right)$$

Substituting into the Black-Scholes PDE with dividend yield $q$:

$$\frac{\partial V}{\partial t}
+ \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2}
+ \left(r - q - \frac{1}{2}\sigma^2\right)\frac{\partial V}{\partial x}
- r V = 0$$

### 9.2 Forward-Time Form

Introducing forward time $\tau = T - t$ (so $\partial/\partial t = -\partial/\partial \tau$):

$$\frac{\partial V}{\partial \tau}
= \underbrace{\frac{1}{2}\sigma^2}_{a}\frac{\partial^2 V}{\partial x^2}
+ \underbrace{\left(r - q - \frac{1}{2}\sigma^2\right)}_{b}\frac{\partial V}{\partial x}
- r V$$

with constant coefficients $a = \frac{1}{2}\sigma^2$ and $b = r - q - \frac{1}{2}\sigma^2$.

The solver starts from the payoff at $\tau = 0$ and steps forward in $\tau$ until $\tau = T$.

### 9.3 Finite-Difference Stencil

Discretise with spatial spacing $\Delta x$ and time step $\Delta\tau$. The central-difference approximations are:

$$\frac{\partial^2 V}{\partial x^2} \approx \frac{V_{i+1} - 2V_i + V_{i-1}}{(\Delta x)^2}, \quad
  \frac{\partial V}{\partial x} \approx \frac{V_{i+1} - V_{i-1}}{2\,\Delta x}$$

Collecting terms, the interior stencil coefficients are:

$$\alpha = \frac{a}{(\Delta x)^2} - \frac{b}{2\,\Delta x} \quad\text{(sub-diagonal)},\quad
  \gamma = \frac{a}{(\Delta x)^2} + \frac{b}{2\,\Delta x} \quad\text{(super-diagonal)}$$
$$\delta = \frac{2a}{(\Delta x)^2} + r \quad\text{(diagonal)}$$

**Crank-Nicolson** averages the explicit ($\theta = 0$) and implicit ($\theta = 1$) steps with $\theta = \frac{1}{2}$:

$$A\,V^{n+1} = B\,V^n$$

where the tridiagonal matrices $A$ (implicit) and $B$ (explicit) have entries:

$$A_{i,i-1} = -\theta\,\Delta\tau\,\alpha, \quad
  A_{i,i}   =  1 + \theta\,\Delta\tau\,\delta, \quad
  A_{i,i+1} = -\theta\,\Delta\tau\,\gamma$$

$$B_{i,i-1} = (1-\theta)\,\Delta\tau\,\alpha, \quad
  B_{i,i}   = 1 - (1-\theta)\,\Delta\tau\,\delta, \quad
  B_{i,i+1} = (1-\theta)\,\Delta\tau\,\gamma$$

At each time step, $V^{n+1}$ is found by solving the tridiagonal system with `scipy.linalg.solve_banded`.

**Rannacher smoothing:** the first few steps use $\theta = 1$ (fully implicit, backward Euler) to damp the oscillations Crank-Nicolson produces near the payoff kink at the strike. Later steps revert to $\theta = \frac{1}{2}$ to recover second-order accuracy.

**American exercise:** after each step, enforce $V_i \geq \max(S_i - K, 0)$ for calls (or $\max(K - S_i, 0)$ for puts) point by point. This reflects the fact that early exercise is always available, so the option can't be worth less than its immediate exercise value.

---

## 10. Put-Call Parity

For two European options on the same underlying with the same strike $K$ and expiry $T$, the following holds exactly under no-arbitrage:

$$\boxed{C - P = S e^{-qT} - K e^{-rT}}$$

**Proof.** Consider two portfolios at time $t$:

- **Portfolio A:** long call, plus cash $K e^{-rT}$ invested at $r$
- **Portfolio B:** long put, plus $e^{-qT}$ units of stock

At expiry, if $S_T > K$: A pays $(S_T - K) + K = S_T$ and B pays $0 + S_T = S_T$. If $S_T \leq K$: A pays $0 + K = K$ and B pays $(K - S_T) + S_T = K$. Both portfolios match in every state, so they must have the same value today. Rearranging gives the result.

Once a call is priced, the put follows immediately. The engine checks parity to machine precision as an internal consistency test.

---

## 11. No-Arbitrage Bounds

European option prices must satisfy the following bounds at all times. Any violation can be turned into a risk-free profit.

**Lower bound (call):**

$$C \geq \max\!\left(S e^{-qT} - K e^{-rT},\ 0\right)$$

If $C < S e^{-qT} - K e^{-rT}$ (with the right-hand side positive), buy the call and invest $K e^{-rT}$ at the risk-free rate. The total cost is less than $S e^{-qT}$. At expiry, the cash grows to $K$. If $S_T > K$, exercise and receive $S_T > 0$; if $S_T \leq K$, the call expires worthless but you keep $K > 0$. Either way there is a net profit, which is a contradiction.

**Lower bound (put):**

$$P \geq \max\!\left(K e^{-rT} - S e^{-qT},\ 0\right)$$

**Upper bounds:**

$$C \leq S e^{-qT}, \quad P \leq K e^{-rT}$$

A call can't be worth more than the dividend-adjusted underlying itself. A put can't be worth more than the discounted strike.

The implied-volatility solver uses these bounds to reject market prices for which no finite volatility exists.

---

## 12. Implied Volatility

The Black-Scholes call price $C = f(\sigma)$ is strictly increasing in $\sigma$ on $(0, \infty)$ and maps onto the interval $(Se^{-qT} - Ke^{-rT},\ Se^{-qT})$. For any market price $C_{\text{mkt}}$ inside that interval there is exactly one **implied volatility** $\hat{\sigma}$ satisfying:

$$f(\hat{\sigma}) = C_{\text{mkt}}$$

### 12.1 Initial Guess

Near the money, $\ln(S/K) \approx 0$, and the call price simplifies to:

$$C \approx S\left(2N\!\left(\frac{\sigma\sqrt{T}}{2}\right) - 1\right)
  \approx S \cdot \frac{\sigma\sqrt{T}}{\sqrt{2\pi}}$$

Inverting gives the Brenner-Subrahmanyam starting point:

$$\hat{\sigma}_0 \approx \sqrt{\frac{2\pi}{T}} \cdot \frac{C}{S}$$

This is accurate at-the-money and reasonable for options that are not too far from it.

### 12.2 Newton-Raphson

Starting from $\sigma_0 = \hat{\sigma}_0$, iterate:

$$\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n) - C_{\text{mkt}}}{\mathcal{V}(\sigma_n)}$$

where $\mathcal{V} = S e^{-qT} \sqrt{T}\, n(d_1)$ is Vega. Newton-Raphson converges quadratically when Vega is large. It struggles for deep in-the-money or out-of-the-money options where Vega is near zero.

### 12.3 Brent Fallback

When Newton-Raphson doesn't converge, the solver falls back to Brent's method on the bracket $[\sigma_{\min}, \sigma_{\max}] = [10^{-8}, 50]$. Brent's method is guaranteed to find a root because $f(\sigma_{\min}) < C_{\text{mkt}} < f(\sigma_{\max})$, so the function changes sign across the bracket.

---

## 13. The SVI Parameterisation

The **Stochastic Volatility Inspired** model (Gatheral, 2004) parameterises the **total implied variance** $w(k) = \hat{\sigma}(k)^2 \cdot T$ as a function of log-moneyness $k = \ln(K/S)$:

$$\boxed{w(k) = a + b\!\left(\rho\,(k - m) + \sqrt{(k - m)^2 + \xi^2}\right)}$$

The five parameters each have a clear geometric meaning:

| Parameter | Meaning |
|-----------|---------|
| $a \geq 0$ | overall level of the smile (ATM total variance) |
| $b \geq 0$ | slope of the wings |
| $\rho \in (-1, 1)$ | skew: negative $\rho$ raises the put wing and depresses the call wing |
| $m$ | horizontal position of the smile minimum |
| $\xi > 0$ | curvature: small $\xi$ gives a sharp V-shape, large $\xi$ a flat U-shape |

### 13.1 No-Arbitrage Constraints

The surface is arbitrage-free in the wings whenever $a, b, \xi \geq 0$ and $|\rho| < 1$. These constraints keep $w(k) \geq 0$ for all $k$: the term under the square root is always at least $\xi^2$, and the $\rho(k-m)$ term can only reduce $w$ by at most $b|\rho||k-m|$, which the square root dominates in the tails.

### 13.2 Analytic Properties

The smile attains its minimum at $k^* = m - \rho\xi/\sqrt{1-\rho^2}$ with value $w(k^*) = a + b\xi\sqrt{1-\rho^2}$.

As $k \to \pm\infty$, $w(k) \sim a + b(1 \pm \rho)|k|$, so the left and right asymptotic slopes are $b(1 - \rho)$ and $b(1 + \rho)$ respectively.

### 13.3 Fitting

Given observed implied volatilities $\hat{\sigma}_i$ at strikes $K_i$, the target total variances are $w_i = \hat{\sigma}_i^2 \cdot T$. The five parameters are found by bounded non-linear least squares:

$$\min_{a,\,b,\,\rho,\,m,\,\xi}\ \sum_{i=1}^n \left(w(k_i;\,a,b,\rho,m,\xi) - w_i\right)^2$$

subject to $a \geq 0$, $b \geq 0$, $\xi \geq 0$, $-0.99 \leq \rho \leq 0.99$.

The starting point is read from the data: $a_0$ from the minimum observed variance, $m_0$ from the strike where it occurs, $b_0$ from the average slope of the smile, $\xi_0$ from the average strike spacing, and $\rho_0 = 0$. The solver is `scipy.optimize.least_squares` with the trust-region-reflective algorithm.
