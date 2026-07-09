#!/usr/bin/env python3
"""
Example: How to compute sensitivities of a SANOS-calibrated surface
         w.r.t. market quotes — without re-calibrating.

Problem:
    You calibrated an arbitrage-free option surface using SANOS (LP).
    Now you need ∂C_fit(K,T)/∂quote_i for risk, hedging, or PnL explain.
    Bump-and-recalibrate: 2N LP solves (slow).
    IFT: 1 LP solve + 1 linear system (fast).

This example:
    1. Creates synthetic market quotes (BS with noise)
    2. Calibrates SANOS surface via LP (their code)
    3. Computes full Jacobian via IFT (our code) — ONE function call
    4. Validates against bump-and-recalibrate
    5. Shows the speedup

Requirements: numpy, scipy
"""

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
import time


# ================================================================
#  Step 0: Black-Scholes helper (for generating test data)
# ================================================================

def bs_call_price(K, vol, sqrtT):
    """BS call price, forward = 1."""
    d1 = -np.log(K) / (vol * sqrtT) + 0.5 * vol * sqrtT
    d2 = d1 - vol * sqrtT
    return norm.cdf(d1) - K * norm.cdf(d2)


# ================================================================
#  Step 1: SANOS LP calibration (simplified, self-contained)
# ================================================================

def calibrate_sanos(strikes, mids, bids, asks):
    """
    Fit arbitrage-free call prices via LP.

    min  Σ |C_i - mid_i|
    s.t. d²C/dK² ≥ 0    (butterfly / positive density)
         dC/dK ≤ 0       (monotone decreasing in strike)
         bid_i ≤ C_i ≤ ask_i

    Returns: fitted_prices, LP_result, A_ub, b_ub, bounds
    """
    N = len(strikes)
    K = strikes

    # Variables: [C_0..C_{N-1}, u_0..u_{N-1}]  (u = |C - mid|)
    c_obj = np.zeros(2 * N)
    c_obj[N:] = 1.0  # minimize sum of u_i

    rows_A, rows_b = [], []

    # |C_i - mid_i| ≤ u_i
    for i in range(N):
        # C_i - u_i ≤ mid_i
        r = np.zeros(2*N); r[i] = 1; r[N+i] = -1
        rows_A.append(r); rows_b.append(mids[i])
        # -C_i - u_i ≤ -mid_i
        r = np.zeros(2*N); r[i] = -1; r[N+i] = -1
        rows_A.append(r); rows_b.append(-mids[i])

    # Butterfly: -d²C/dK² ≤ 0
    for i in range(1, N-1):
        dK_L = K[i] - K[i-1]
        dK_R = K[i+1] - K[i]
        r = np.zeros(2*N)
        r[i-1] = -1/dK_L; r[i] = 1/dK_L + 1/dK_R; r[i+1] = -1/dK_R
        rows_A.append(r); rows_b.append(0.0)

    # Monotone: C_{i+1} - C_i ≤ 0
    for i in range(N-1):
        r = np.zeros(2*N); r[i] = -1; r[i+1] = 1
        rows_A.append(r); rows_b.append(0.0)

    A_ub = np.array(rows_A)
    b_ub = np.array(rows_b)
    bounds = [(bids[i], asks[i]) for i in range(N)] + [(0, None)] * N

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    assert result.success, f"LP failed: {result.message}"

    return result.x[:N], result, A_ub, b_ub, bounds


# ================================================================
#  Step 2: IFT sensitivity (the key function)
# ================================================================

def sanos_sensitivity(A_ub, b_ub, bounds, lp_result, n_prices, tol=1e-7):
    """
    Compute ∂(fitted prices)/∂(mid quotes) via IFT.

    After LP calibration, call this once to get the full Jacobian.
    No re-solving needed.

    Args:
        A_ub, b_ub: constraint matrix/RHS from the LP
        bounds: variable bounds from the LP
        lp_result: scipy.optimize.OptimizeResult
        n_prices: number of price variables (first n_prices of x)

    Returns:
        dC_dmid: (n_prices, n_prices) Jacobian
                 dC_dmid[i,j] = ∂C_fit(K_i) / ∂mid(K_j)
    """
    x = lp_result.x
    n_vars = len(x)

    # Find active constraints
    slack = b_ub - A_ub @ x
    active_ineq = np.abs(slack) < tol
    active_lb = np.array([abs(x[i] - bounds[i][0]) < tol
                          if bounds[i][0] is not None else False
                          for i in range(n_vars)])
    active_ub = np.array([abs(x[i] - bounds[i][1]) < tol
                          if bounds[i][1] is not None else False
                          for i in range(n_vars)])

    # Build active constraint matrix
    rows = []
    mid_map = []  # which mid does this row's RHS depend on? (idx, sign) or None

    for k in range(len(b_ub)):
        if active_ineq[k]:
            rows.append(A_ub[k])
            # First 2*n_prices rows of b_ub contain mid dependencies:
            #   row 2j:   b = +mid[j]
            #   row 2j+1: b = -mid[j]
            if k < 2 * n_prices:
                mid_map.append((k // 2, 1.0 if k % 2 == 0 else -1.0))
            else:
                mid_map.append(None)

    for i in range(n_vars):
        if active_lb[i]:
            r = np.zeros(n_vars); r[i] = -1
            rows.append(r)
            mid_map.append(None)
    for i in range(n_vars):
        if active_ub[i]:
            r = np.zeros(n_vars); r[i] = 1
            rows.append(r)
            mid_map.append(None)

    A_act = np.array(rows)
    n_act = len(rows)

    # Solve A_act @ dx = I
    if n_act == n_vars:
        inv_A = np.linalg.solve(A_act, np.eye(n_act))
    else:
        inv_A = np.linalg.lstsq(A_act, np.eye(n_act), rcond=None)[0]

    # Chain rule: ∂C_fit/∂mid via ∂b/∂mid
    dC_dmid = np.zeros((n_prices, n_prices))
    for col, dep in enumerate(mid_map):
        if dep is not None:
            j, sign = dep
            dC_dmid[:, j] += inv_A[:n_prices, col] * sign

    return dC_dmid


# ================================================================
#  Example
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SANOS + IFT: Surface Sensitivity Without Re-Calibration")
    print("=" * 65)

    # --- Generate market data with noise (butterfly violations) ---
    np.random.seed(42)
    K = np.linspace(0.75, 1.25, 15)     # strikes (normalized, forward=1)
    vol_true = 0.20
    sqrtT = 1.0                          # 1-year expiry

    mid_true = bs_call_price(K, vol_true, sqrtT)
    noise = np.random.randn(len(K)) * 0.005   # 50bp noise — creates more arb violations
    mids = np.clip(mid_true + noise, np.maximum(1-K, 0) + 1e-4, 1 - 1e-4)

    spread = 0.008  # 80bp bid-ask (wider to give LP room)
    bids = np.maximum(mids - spread, np.maximum(1-K, 0) + 1e-8)
    asks = np.minimum(mids + spread, 1 - 1e-8)
    bids = np.minimum(bids, asks - 1e-6)

    N = len(K)
    print(f"\nMarket: {N} quotes, vol≈{vol_true}, T={sqrtT**2}")
    print(f"  Noise: 50bp,  Spread: 80bp")

    # --- Calibrate ---
    t0 = time.time()
    C_fit, lp_res, A_ub, b_ub, bounds = calibrate_sanos(K, mids, bids, asks)
    t_calib = time.time() - t0

    max_dev = np.max(np.abs(C_fit - mids))
    print(f"\n1. Calibration ({t_calib*1000:.1f}ms):")
    print(f"   Max |fit - mid| = {max_dev:.4f}")

    # Check arb-free
    dK = K[1:] - K[:-1]
    d2CdK2 = np.diff(np.diff(C_fit)/dK) / (0.5*(dK[:-1]+dK[1:]))
    print(f"   Butterfly min d²C/dK² = {d2CdK2.min():.2e} (≥0 = OK)")

    # --- IFT: one call ---
    t0 = time.time()
    J = sanos_sensitivity(A_ub, b_ub, bounds, lp_res, N)
    t_ift = time.time() - t0

    print(f"\n2. IFT Jacobian ({t_ift*1000:.1f}ms):")
    print(f"   Shape: {J.shape}  (∂C_fit[i] / ∂mid[j])")

    # Show diagonal
    print(f"\n   Diagonal (self-sensitivity):")
    print(f"   {'Strike':>8s}  {'∂C/∂mid':>8s}  {'Meaning':>30s}")
    for i in range(N):
        v = J[i, i]
        if abs(v - 1.0) < 0.01:
            meaning = "passes through (no binding arb)"
        elif abs(v) < 0.01:
            meaning = "pinned by arb constraint"
        else:
            meaning = f"partial: {v:.0%} from mid, rest from arb"
        print(f"   {K[i]:>8.3f}  {v:>8.4f}  {meaning}")

    # --- Validate against bump ---
    print(f"\n3. Validation (bump-and-recalibrate):")
    h = 1e-6
    max_err = 0
    for j in [0, N//4, N//2, 3*N//4, N-1]:
        m_u = mids.copy(); m_u[j] += h; b_u = bids.copy(); b_u[j] += h; a_u = asks.copy(); a_u[j] += h
        c_u, *_ = calibrate_sanos(K, m_u, b_u, a_u)
        m_d = mids.copy(); m_d[j] -= h; b_d = bids.copy(); b_d[j] -= h; a_d = asks.copy(); a_d[j] -= h
        c_d, *_ = calibrate_sanos(K, m_d, b_d, a_d)
        fd = (c_u - c_d) / (2*h)
        err = np.max(np.abs(J[:, j] - fd))
        max_err = max(max_err, err)
        print(f"   Quote j={j:2d} (K={K[j]:.3f}): max|IFT−FD| = {err:.1e}")
    print(f"   Overall max error: {max_err:.1e}")

    # --- Benchmark ---
    t0 = time.time()
    for j in range(N):
        for s in [+1, -1]:
            m = mids.copy(); m[j] += s*h
            b = bids.copy(); b[j] += s*h
            a = asks.copy(); a[j] += s*h
            calibrate_sanos(K, m, b, a)
    t_bump = time.time() - t0

    print(f"\n4. Benchmark ({N} quotes):")
    print(f"   Bump: {t_bump:.3f}s  ({2*N} LP solves)")
    print(f"   IFT:  {t_ift*1000:.1f}ms     (1 linear solve)")
    print(f"   Speedup: {t_bump/t_ift:.0f}x")

    # --- Use case: which quote matters most for ATM price? ---
    atm = N // 2
    print(f"\n5. Use case: which quote most affects ATM price (K={K[atm]:.3f})?")
    ranking = np.argsort(np.abs(J[atm]))[::-1]
    for rank, j in enumerate(ranking[:5]):
        print(f"   #{rank+1}: quote K={K[j]:.3f}, sensitivity = {J[atm,j]:+.4f}")

    # --- Cross-sensitivity: how does bumping one strike affect others? ---
    print(f"\n6. Cross-sensitivity (off-diagonal):")
    print(f"   How bumping K=0.893 quote affects fitted prices at OTHER strikes:")
    j_bump = 4  # K=0.893
    for i in range(N):
        if abs(J[i, j_bump]) > 0.001:
            print(f"   K={K[i]:.3f}: ∂C_fit/∂mid = {J[i, j_bump]:+.4f}")

    print(f"\n{'='*65}")
    print(f"  Done.  IFT gives exact Jacobian from 1 linear solve.")
    print(f"  No re-calibration. No finite differences. No AADC needed.")
    print(f"{'='*65}")
