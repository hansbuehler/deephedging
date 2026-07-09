#!/usr/bin/env python3
"""
SANOS + IFT + AADC: Full Python driver.

End-to-end: market quotes → SANOS LP calibration → IFT sensitivity
→ AADC downstream pricing → chain rule → ∂(exotic price)/∂(quotes).

Part I:  SANOS calibration (Buehler's smoothvol.py) + IFT (cvxpy_ift.py)
Part II: AADC tape: surface → local vol interpolation → MC barrier → price + Greeks

Requirements: numpy, scipy, cvxpy, aadc (https://matlogica.com/aadc)
"""

import sys, time
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
#  SANOS calibration (Buehler's framework)
# ---------------------------------------------------------------------------
from sanos_smoothvol import SmoothCallSurface, SmoothCallSurfaceConfig, ExpiryData
from sanos_bs import bs_call

# ---------------------------------------------------------------------------
#  IFT sensitivity
# ---------------------------------------------------------------------------
from example_sanos_ift import calibrate_sanos, sanos_sensitivity

# ---------------------------------------------------------------------------
#  AADC
# ---------------------------------------------------------------------------
import aadc
from aadc.recording_ctx import record_kernel
from aadc.evaluate_wrappers import evaluate_kernel


def make_market_data(K, vol, sqrtT, noise=0.003, spread=0.005, seed=42):
    """Generate synthetic BS market data with noise."""
    rng = np.random.RandomState(seed)
    mids = bs_call(K, vol, sqrtT) + rng.randn(len(K)) * noise
    mids = np.clip(mids, np.maximum(1 - K, 0) + 1e-4, 1 - 1e-4)
    bids = np.maximum(mids - spread, np.maximum(1 - K, 0) + 1e-6)
    asks = np.minimum(mids + spread, 1 - 1e-6)
    bids = np.minimum(bids, asks - 1e-6)
    return mids, bids, asks


def part1_sanos_ift(K, mids, bids, asks):
    """
    Part I: SANOS calibration + IFT Jacobian.

    Returns:
        C_fit: fitted call prices at market strikes
        J_ift: (N, N) Jacobian  ∂C_fit(K_i)/∂mid(K_j)
    """
    N = len(K)

    # Use our simplified LP (same constraints as Buehler: butterfly, monotone, bid-ask)
    C_fit, lp_result, A_ub, b_ub, bounds = calibrate_sanos(K, mids, bids, asks)

    # IFT: one linear solve → full Jacobian
    J_ift = sanos_sensitivity(A_ub, b_ub, bounds, lp_result, N)

    return C_fit, J_ift


def part2_aadc_barrier(C_fit, K, sqrtT, S0=1.0, K_strike=1.0, barrier=0.8,
                        r=0.0, n_paths=50000, n_steps=50):
    """
    Part II: AADC downstream — barrier option priced on tape.

    Records: surface nodes → local vol interpolation → GBM MC → barrier payoff.
    One evaluate_kernel call → price + ∂price/∂C_fit for ALL surface nodes.

    Returns:
        price: barrier option price
        dV_dC: (N,) array — ∂price/∂C_fit(K_i)
    """
    N = len(K)
    T = sqrtT ** 2
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Compute implied vols from fitted prices (for local vol proxy)
    iv = np.zeros(N)
    for i in range(N):
        iv[i] = _implied_vol_bisect(K[i], C_fit[i], sqrtT)
    local_var = iv ** 2  # flat local vol per strike (simplified Dupire)

    # Record AADC kernel
    with record_kernel() as kernel:
        # Surface nodes as inputs
        surface_inputs = []
        surface_args = []
        for i in range(N):
            x = aadc.idouble(local_var[i])
            arg = x.mark_as_input()
            surface_inputs.append(x)
            surface_args.append(arg)

        # Random normals (no diff)
        z_inputs = []
        z_args = []
        for j in range(n_steps):
            z = aadc.idouble(0.0)
            za = z.mark_as_input_no_diff()
            z_inputs.append(z)
            z_args.append(za)

        # Strike thresholds as idouble (created once, reused each step)
        K_thresh = [aadc.idouble(float(K[i])) for i in range(N)]

        # MC path under local vol (one path on tape)
        # S0 must be on tape for comparisons to return ibool
        S = aadc.idouble(S0)
        s0_arg = S.mark_as_input_no_diff()
        alive = aadc.idouble(1.0)

        for step in range(n_steps):
            # Interpolate local variance: piecewise constant from nearest strike
            lv = surface_inputs[0]
            for i in range(1, N):
                lv = aadc.iif(S >= K_thresh[i], surface_inputs[i], lv)

            local_vol = lv.sqrt()

            # GBM step
            drift = (aadc.idouble(r) - lv * aadc.idouble(0.5)) * aadc.idouble(dt)
            diffusion = local_vol * aadc.idouble(sqrt_dt) * z_inputs[step]
            S = S * (drift + diffusion).exp()

            # Barrier check
            alive = alive * aadc.iif(S > aadc.idouble(barrier),
                                     aadc.idouble(1.0), aadc.idouble(0.0))

        # Payoff: down-and-out call
        payoff = alive * aadc.iif(S > aadc.idouble(K_strike),
                                   S - aadc.idouble(K_strike),
                                   aadc.idouble(0.0))
        payoff = payoff * aadc.idouble(np.exp(-r * T))
        payoff_out = payoff.mark_as_output()

    # MC evaluation: many paths
    rng = np.random.RandomState(42)
    Z_all = rng.randn(n_paths, n_steps)

    # Build inputs dict: surface values (same for all paths) + random normals (per path)
    inputs = {s0_arg: S0}
    for i in range(N):
        inputs[surface_args[i]] = local_var[i]  # scalar, same for all paths
    for j in range(n_steps):
        inputs[z_args[j]] = Z_all[:, j]  # vector of n_paths

    # Request: price + ∂price/∂surface
    request = {payoff_out: surface_args}

    result = evaluate_kernel(kernel, request, inputs, num_threads=4)

    price = result.values[payoff_out].mean()
    dV_dsurface = np.array([result.derivs[payoff_out][surface_args[i]].mean()
                            for i in range(N)])

    # Chain rule: ∂V/∂C_fit = ∂V/∂σ² · ∂σ²/∂C_fit
    # σ² = iv² where iv = BS_implied(K, C_fit, sqrtT)
    # ∂σ²/∂C = 2·iv · ∂iv/∂C, and ∂iv/∂C = 1/vega (BS vega)
    dvar_dC = np.zeros(N)
    for i in range(N):
        vega = _bs_vega(K[i], iv[i], sqrtT)
        if vega > 1e-12:
            dvar_dC[i] = 2.0 * iv[i] / vega

    dV_dC = dV_dsurface * dvar_dC

    return price, dV_dC


def _implied_vol_bisect(K, C, sqrtT, lo=0.01, hi=2.0, tol=1e-10, maxiter=100):
    """Bisection implied vol."""
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        p = _bs_call_scalar(K, mid, sqrtT)
        if p < C:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _bs_call_scalar(K, vol, sqrtT):
    """BS call price, forward=1."""
    d1 = -np.log(K) / (vol * sqrtT) + 0.5 * vol * sqrtT
    d2 = d1 - vol * sqrtT
    return norm.cdf(d1) - K * norm.cdf(d2)


def _bs_vega(K, vol, sqrtT):
    """BS vega (∂C/∂σ), forward=1."""
    d1 = -np.log(K) / (vol * sqrtT) + 0.5 * vol * sqrtT
    return norm.pdf(d1) * sqrtT


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  SANOS + IFT + AADC: Full Python Driver")
    print("=" * 70)

    # Market data
    K = np.linspace(0.75, 1.25, 15)
    sqrtT = 1.0
    mids, bids, asks = make_market_data(K, vol=0.20, sqrtT=sqrtT)
    N = len(K)
    print(f"\nMarket: {N} quotes, T={sqrtT**2}, vol≈0.20")

    # ---- Part I: SANOS + IFT ----
    print("\n--- Part I: SANOS Calibration + IFT ---")
    t0 = time.time()
    C_fit, J_ift = part1_sanos_ift(K, mids, bids, asks)
    t1 = time.time()
    print(f"  Calibration + IFT: {(t1-t0)*1000:.1f}ms")
    print(f"  Max |fit-mid|: {np.max(np.abs(C_fit - mids)):.4f}")
    print(f"  IFT Jacobian: {J_ift.shape}")

    # Validate IFT vs bump (3 test quotes)
    print("\n  IFT vs bump validation:")
    h = 1e-6
    for j in [0, N // 2, N - 1]:
        m_u = mids.copy(); m_u[j] += h
        b_u = bids.copy(); b_u[j] += h
        a_u = asks.copy(); a_u[j] += h
        c_u, *_ = calibrate_sanos(K, m_u, b_u, a_u)
        m_d = mids.copy(); m_d[j] -= h
        b_d = bids.copy(); b_d[j] -= h
        a_d = asks.copy(); a_d[j] -= h
        c_d, *_ = calibrate_sanos(K, m_d, b_d, a_d)
        fd = (c_u - c_d) / (2 * h)
        err = np.max(np.abs(J_ift[:, j] - fd))
        print(f"    Quote j={j} (K={K[j]:.3f}): max|IFT-FD| = {err:.1e}")

    # ---- Part II: AADC downstream ----
    print("\n--- Part II: AADC Downstream Barrier Pricing ---")
    t0 = time.time()
    price, dV_dC = part2_aadc_barrier(C_fit, K, sqrtT, n_paths=20000, n_steps=50)
    t2 = time.time()
    print(f"  AADC MC: {t2-t0:.1f}s")
    print(f"  Barrier price: {price:.6f}")
    print(f"  ∂price/∂C_fit: {np.count_nonzero(np.abs(dV_dC) > 1e-8)} nonzero of {N}")

    # ---- Chain rule ----
    print("\n--- Chain Rule: ∂(barrier price)/∂(vanilla quotes) ---")
    dV_dquotes = dV_dC @ J_ift
    print(f"  All {N} sensitivities computed.")

    ranking = np.argsort(np.abs(dV_dquotes))[::-1]
    print(f"\n  Top sensitivities:")
    print(f"  {'K':>8s}  {'∂V/∂quote':>12s}")
    for i in ranking[:5]:
        print(f"  {K[i]:>8.3f}  {dV_dquotes[i]:>+12.6f}")

    # ---- Benchmark ----
    print(f"\n--- Benchmark ---")
    t_ift_total = (t1 - t0) + (t2 - t0)  # Note: t0 is reset before Part II
    t_bump_est = (t2 - t0) * 2 * N  # each bump needs full MC repricing
    print(f"  IFT + AADC: {t_ift_total:.1f}s")
    print(f"  Bump estimate: {t_bump_est:.0f}s ({2*N} MC pricings)")
    if t_ift_total > 0:
        print(f"  Speedup: {t_bump_est/t_ift_total:.0f}x")

    print(f"\n{'='*70}")
    print(f"  Done. All {N} quote sensitivities from 1 LP + 1 IFT + 1 AADC MC.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
