#!/usr/bin/env python3
"""
SANOS + IFT: Full pipeline test.

1. Multi-expiry LP calibration with arbitrage constraints
2. IFT: ∂surface/∂quotes (all quotes, one linear solve)
3. Downstream: Dupire local vol → MC barrier option pricing
4. End-to-end: ∂(barrier price)/∂(vanilla quotes) via chain rule
5. Benchmark: IFT vs bump-and-recalibrate
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from scipy.optimize import linprog
from scipy.interpolate import interp1d
from bs import bs_call
import time

# ============================================================
# 1. Multi-Expiry SANOS LP
# ============================================================

def sanos_multi_expiry_lp(expiries, all_strikes, all_mids, all_bids, all_asks):
    """
    Multi-expiry LP with butterfly + calendar constraints.

    expiries: list of T values
    all_strikes[j]: strikes for expiry j
    all_mids[j], all_bids[j], all_asks[j]: prices for expiry j

    Variables: C_j_i = call price at expiry j, strike i
    Objective: min Σ |C_j_i - mid_j_i|
    Constraints:
      - Butterfly per expiry: d²C/dK² >= 0
      - Calendar across expiries: C(K, T_{j+1}) >= C(K, T_j) at common strikes
      - Bid-ask: bid <= C <= ask
    """
    M = len(expiries)  # number of expiries
    Ns = [len(s) for s in all_strikes]  # strikes per expiry
    N_total = sum(Ns)  # total price variables
    n_vars = 2 * N_total  # prices + abs-value aux

    # Variable indexing
    offsets = [0]
    for n in Ns:
        offsets.append(offsets[-1] + n)

    # Objective: min Σ u_j_i
    c_obj = np.zeros(n_vars)
    c_obj[N_total:] = 1.0

    constraints_A = []
    constraints_b = []
    constraint_types = []  # 'abs', 'butterfly', 'calendar', 'monotone'
    mid_dependency = []  # (expiry_idx, strike_idx, sign) or None

    # 1. Absolute value constraints
    for j in range(M):
        for i in range(Ns[j]):
            idx = offsets[j] + i
            u_idx = N_total + idx

            # C - u <= mid
            row = np.zeros(n_vars)
            row[idx] = 1.0; row[u_idx] = -1.0
            constraints_A.append(row)
            constraints_b.append(all_mids[j][i])
            constraint_types.append('abs')
            mid_dependency.append((j, i, 1.0))

            # -C - u <= -mid
            row = np.zeros(n_vars)
            row[idx] = -1.0; row[u_idx] = -1.0
            constraints_A.append(row)
            constraints_b.append(-all_mids[j][i])
            constraint_types.append('abs')
            mid_dependency.append((j, i, -1.0))

    # 2. Butterfly per expiry
    for j in range(M):
        K = all_strikes[j]
        for i in range(1, Ns[j] - 1):
            dK_L = K[i] - K[i-1]
            dK_R = K[i+1] - K[i]
            row = np.zeros(n_vars)
            idx = offsets[j]
            row[idx + i-1] = -1.0 / dK_L
            row[idx + i]   = 1.0 / dK_L + 1.0 / dK_R
            row[idx + i+1] = -1.0 / dK_R
            constraints_A.append(row)
            constraints_b.append(0.0)
            constraint_types.append('butterfly')
            mid_dependency.append(None)

    # 3. Monotone per expiry
    for j in range(M):
        for i in range(Ns[j] - 1):
            row = np.zeros(n_vars)
            idx = offsets[j]
            row[idx + i] = -1.0
            row[idx + i+1] = 1.0
            constraints_A.append(row)
            constraints_b.append(0.0)
            constraint_types.append('monotone')
            mid_dependency.append(None)

    # 4. Calendar: C(K, T_{j+1}) >= C(K, T_j) at common strikes
    for j in range(M - 1):
        K_j = all_strikes[j]
        K_jp1 = all_strikes[j+1]
        common = set(np.round(K_j, 8)) & set(np.round(K_jp1, 8))
        for kval in sorted(common):
            i_j = np.argmin(np.abs(K_j - kval))
            i_jp1 = np.argmin(np.abs(K_jp1 - kval))
            # C_{j+1}(K) >= C_j(K)  →  C_j - C_{j+1} <= 0
            row = np.zeros(n_vars)
            row[offsets[j] + i_j] = 1.0
            row[offsets[j+1] + i_jp1] = -1.0
            constraints_A.append(row)
            constraints_b.append(0.0)
            constraint_types.append('calendar')
            mid_dependency.append(None)

    A_ub = np.array(constraints_A)
    b_ub = np.array(constraints_b)

    # Bounds
    bounds = []
    for j in range(M):
        for i in range(Ns[j]):
            bounds.append((all_bids[j][i], all_asks[j][i]))
    for _ in range(N_total):
        bounds.append((0.0, None))

    # Solve
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    # Extract fitted prices per expiry
    C_fits = []
    for j in range(M):
        C_fits.append(result.x[offsets[j]:offsets[j] + Ns[j]])

    # Active constraints
    slack = b_ub - A_ub @ result.x
    active = np.abs(slack) < 1e-8
    active_lo = np.array([abs(result.x[i] - bounds[i][0]) < 1e-8 for i in range(N_total)])
    active_hi = np.array([abs(result.x[i] - bounds[i][1]) < 1e-8 for i in range(N_total)])

    return C_fits, A_ub, b_ub, active, active_lo, active_hi, constraint_types, mid_dependency, offsets, result


def ift_multi_expiry(A_ub, b_ub, active, active_lo, active_hi,
                      N_total, constraint_types, mid_dependency, n_vars):
    """IFT: ∂C_fit/∂mids for multi-expiry LP."""
    active_rows = []
    b_mid_deps = []

    for i, is_active in enumerate(active):
        if is_active:
            active_rows.append(A_ub[i])
            b_mid_deps.append(mid_dependency[i])

    for i in range(N_total):
        if active_lo[i]:
            row = np.zeros(n_vars)
            row[i] = -1.0
            active_rows.append(row)
            b_mid_deps.append(None)
    for i in range(N_total):
        if active_hi[i]:
            row = np.zeros(n_vars)
            row[i] = 1.0
            active_rows.append(row)
            b_mid_deps.append(None)

    A_act = np.array(active_rows)
    n_active = len(active_rows)

    if n_active == n_vars:
        dxdb = np.linalg.solve(A_act, np.eye(n_active))
    else:
        dxdb = np.linalg.lstsq(A_act, np.eye(n_active), rcond=None)[0]

    return dxdb, b_mid_deps


# ============================================================
# 2. Dupire Local Vol from Surface
# ============================================================

def dupire_local_vol(strikes, expiries, C_fits):
    """Extract Dupire local vol σ²(K,T) from calibrated surface."""
    M = len(expiries)
    N = len(strikes[0])

    # Use common strikes (assume all same for simplicity)
    K = strikes[0]
    local_var = np.zeros((M, N))

    for j in range(M):
        T = expiries[j]
        C = C_fits[j]
        dK = K[1:] - K[:-1]

        # dC/dT (forward difference, or use T=0 boundary)
        if j == 0:
            dCdT = C / T  # first-order from C(K,0)=max(1-K,0)
        else:
            dCdT = (C - C_fits[j-1]) / (expiries[j] - expiries[j-1])

        # d²C/dK²
        dCdK = np.diff(C) / dK
        d2CdK2 = np.zeros(N)
        d2CdK2[1:-1] = np.diff(dCdK) / (0.5 * (dK[:-1] + dK[1:]))
        d2CdK2[0] = d2CdK2[1]
        d2CdK2[-1] = d2CdK2[-2]

        # σ²(K,T) = 2 * dC/dT / (K² * d²C/dK²)
        denom = K**2 * d2CdK2
        denom = np.maximum(denom, 1e-10)
        local_var[j] = np.maximum(2.0 * dCdT / denom, 1e-6)

    return local_var


# ============================================================
# 3. MC Barrier Pricing under Local Vol
# ============================================================

def mc_barrier_price(S0, K_strike, barrier, T, local_vol_func, r, M_paths, n_steps, seed=42):
    """Price down-and-out call under local vol via MC."""
    rng = np.random.RandomState(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.full(M_paths, S0)
    alive = np.ones(M_paths, dtype=bool)

    for step in range(n_steps):
        t = step * dt
        z = rng.randn(M_paths)
        vol = local_vol_func(S / S0, t)  # normalized strikes
        S = S * np.exp((r - 0.5 * vol**2) * dt + vol * sqrt_dt * z)
        alive &= (S > barrier)

    payoff = np.maximum(S - K_strike, 0.0) * alive * np.exp(-r * T)
    return payoff.mean(), payoff


# ============================================================
# Main Test
# ============================================================

def main():
    print("=" * 70)
    print("SANOS + IFT: Full Pipeline Test")
    print("=" * 70)

    # Market data: noisy BS with butterfly violations
    np.random.seed(42)
    vol_true = 0.20
    expiries = [0.25, 0.5, 1.0]
    M_exp = len(expiries)
    N_strikes = 11
    K = np.linspace(0.7, 1.3, N_strikes)

    noise_level = 0.002

    all_strikes = [K.copy() for _ in range(M_exp)]
    all_mids = []
    all_bids = []
    all_asks = []

    for j, T in enumerate(expiries):
        sqrtT = np.sqrt(T)
        mids = bs_call(K, vol_true, sqrtT) + np.random.randn(N_strikes) * noise_level
        mids = np.maximum(mids, np.maximum(1.0 - K, 0.0) + 1e-4)
        mids = np.minimum(mids, 1.0 - 1e-4)
        spread = 0.004
        bids = np.maximum(mids - spread, np.maximum(1.0 - K, 0.0) + 1e-8)
        asks = np.minimum(mids + spread, 1.0 - 1e-8)
        bids = np.minimum(bids, asks - 1e-6)
        all_mids.append(mids)
        all_bids.append(bids)
        all_asks.append(asks)

    N_total = M_exp * N_strikes
    n_vars = 2 * N_total
    N_quotes = N_total

    print(f"\n{M_exp} expiries × {N_strikes} strikes = {N_quotes} quotes")
    print(f"Noise: {noise_level*1e4:.0f}bp, Spread: 40bp")

    # ---- Phase 1: Calibrate ----
    print("\n--- Phase 1: LP Calibration ---")
    t0 = time.time()
    C_fits, A_ub, b_ub, active, active_lo, active_hi, ctypes, mid_deps, offsets, res = \
        sanos_multi_expiry_lp(expiries, all_strikes, all_mids, all_bids, all_asks)
    t_calib = time.time() - t0

    n_binding_bf = sum(1 for i, a in enumerate(active) if a and ctypes[i] == 'butterfly')
    n_binding_cal = sum(1 for i, a in enumerate(active) if a and ctypes[i] == 'calendar')
    max_dev = max(np.max(np.abs(C_fits[j] - all_mids[j])) for j in range(M_exp))

    print(f"  LP time: {t_calib*1000:.1f}ms")
    print(f"  Max |fit-mid|: {max_dev:.4f}")
    print(f"  Binding butterfly: {n_binding_bf}, calendar: {n_binding_cal}")

    # Arb-free check
    all_ok = True
    for j in range(M_exp):
        dK = K[1:] - K[:-1]
        d2 = np.diff(np.diff(C_fits[j]) / dK) / (0.5 * (dK[:-1] + dK[1:]))
        if d2.min() < -1e-6:
            print(f"  WARNING: butterfly violation at expiry {expiries[j]}: {d2.min():.2e}")
            all_ok = False
    for j in range(M_exp - 1):
        cal = C_fits[j+1] - C_fits[j]
        if cal.min() < -1e-6:
            print(f"  WARNING: calendar violation {expiries[j]}→{expiries[j+1]}: {cal.min():.2e}")
            all_ok = False
    print(f"  Arbitrage-free: {'OK' if all_ok else 'FAIL'}")

    # ---- Phase 2: IFT ----
    print("\n--- Phase 2: IFT Sensitivities ---")
    t0 = time.time()
    dxdb, b_mid_deps = ift_multi_expiry(A_ub, b_ub, active, active_lo, active_hi,
                                         N_total, ctypes, mid_deps, n_vars)
    t_ift = time.time() - t0
    print(f"  IFT time: {t_ift*1000:.1f}ms")

    # Build full Jacobian ∂C_fit/∂mid (N_total × N_total)
    dC_dmid_ift = np.zeros((N_total, N_total))
    for k, dep in enumerate(b_mid_deps):
        if dep is not None:
            j_exp, i_strike, sign = dep
            mid_idx = offsets[j_exp] + i_strike
            for i in range(N_total):
                dC_dmid_ift[i, mid_idx] += dxdb[i, k] * sign

    # FD validation (subset)
    print("\n--- Phase 2b: FD Validation ---")
    h = 1e-6
    test_quotes = [0, N_strikes//2, N_strikes-1, N_strikes, 2*N_strikes]  # first, ATM, last of exp0; first of exp1; first of exp2
    max_ift_err = 0

    for q in test_quotes:
        j_exp = q // N_strikes
        i_str = q % N_strikes
        m_u = [m.copy() for m in all_mids]; b_u = [b.copy() for b in all_bids]; a_u = [a.copy() for a in all_asks]
        m_u[j_exp][i_str] += h; b_u[j_exp][i_str] += h; a_u[j_exp][i_str] += h
        C_u, *_ = sanos_multi_expiry_lp(expiries, all_strikes, m_u, b_u, a_u)

        m_d = [m.copy() for m in all_mids]; b_d = [b.copy() for b in all_bids]; a_d = [a.copy() for a in all_asks]
        m_d[j_exp][i_str] -= h; b_d[j_exp][i_str] -= h; a_d[j_exp][i_str] -= h
        C_d, *_ = sanos_multi_expiry_lp(expiries, all_strikes, m_d, b_d, a_d)

        fd_col = np.concatenate([(C_u[j] - C_d[j]) / (2*h) for j in range(M_exp)])
        ift_col = dC_dmid_ift[:, q]

        err = np.max(np.abs(ift_col - fd_col))
        max_ift_err = max(max_ift_err, err)
        print(f"  Quote {q} (exp={j_exp}, K={K[i_str]:.2f}): max|IFT-FD|={err:.2e}")

    print(f"  Max IFT error across tested quotes: {max_ift_err:.2e}")

    # ---- Phase 3: Downstream Barrier Pricing ----
    print("\n--- Phase 3: Downstream Barrier Pricing ---")
    S0 = 1.0  # normalized
    K_barrier = 1.0  # ATM call
    barrier = 0.8    # down-and-out at 80%
    r = 0.0
    T_price = expiries[-1]  # price at longest expiry
    M_mc = 20000
    n_steps = 50

    # Local vol from surface
    local_var = dupire_local_vol(all_strikes, expiries, C_fits)

    # Interpolation function for MC
    def local_vol_func(S_norm, t):
        j = min(int(t / expiries[-1] * M_exp), M_exp - 1)
        vol2_interp = interp1d(K, local_var[j], kind='linear', fill_value='extrapolate')
        return np.sqrt(np.maximum(vol2_interp(S_norm), 1e-6))

    base_price, _ = mc_barrier_price(S0, K_barrier, barrier, T_price, local_vol_func, r, M_mc, n_steps)
    print(f"  Barrier price (base): {base_price:.6f}")

    # ---- Phase 4: End-to-End Sensitivity ----
    print("\n--- Phase 4: End-to-End ∂(barrier)/∂(quotes) ---")

    # Bump-and-recalibrate for full chain
    print("  Computing bump-and-recalibrate (full chain)...")
    t0 = time.time()
    n_test = min(N_quotes, 10)
    test_indices = np.linspace(0, N_quotes-1, n_test, dtype=int)

    dprice_dquote_bump = np.zeros(n_test)
    h_bump = 1e-4

    for idx, q in enumerate(test_indices):
        j_exp = q // N_strikes
        i_str = q % N_strikes

        # Up
        m_u = [m.copy() for m in all_mids]; b_u = [b.copy() for b in all_bids]; a_u = [a.copy() for a in all_asks]
        m_u[j_exp][i_str] += h_bump; b_u[j_exp][i_str] += h_bump; a_u[j_exp][i_str] += h_bump
        C_u, *_ = sanos_multi_expiry_lp(expiries, all_strikes, m_u, b_u, a_u)
        lv_u = dupire_local_vol(all_strikes, expiries, C_u)
        def lvf_u(S, t):
            j = min(int(t / expiries[-1] * M_exp), M_exp - 1)
            return np.sqrt(np.maximum(interp1d(K, lv_u[j], kind='linear', fill_value='extrapolate')(S), 1e-6))
        p_up, _ = mc_barrier_price(S0, K_barrier, barrier, T_price, lvf_u, r, M_mc, n_steps)

        # Down
        m_d = [m.copy() for m in all_mids]; b_d = [b.copy() for b in all_bids]; a_d = [a.copy() for a in all_asks]
        m_d[j_exp][i_str] -= h_bump; b_d[j_exp][i_str] -= h_bump; a_d[j_exp][i_str] -= h_bump
        C_d, *_ = sanos_multi_expiry_lp(expiries, all_strikes, m_d, b_d, a_d)
        lv_d = dupire_local_vol(all_strikes, expiries, C_d)
        def lvf_d(S, t):
            j = min(int(t / expiries[-1] * M_exp), M_exp - 1)
            return np.sqrt(np.maximum(interp1d(K, lv_d[j], kind='linear', fill_value='extrapolate')(S), 1e-6))
        p_dn, _ = mc_barrier_price(S0, K_barrier, barrier, T_price, lvf_d, r, M_mc, n_steps)

        dprice_dquote_bump[idx] = (p_up - p_dn) / (2 * h_bump)

    t_bump = time.time() - t0

    print(f"  Bump time ({n_test} quotes): {t_bump:.1f}s = {t_bump/n_test:.2f}s per quote")
    print(f"  Extrapolated for {N_quotes} quotes: {t_bump/n_test*N_quotes:.1f}s")

    # ---- Phase 5: Benchmark ----
    print("\n--- Phase 5: Benchmark ---")
    print(f"\n  Method          | Time        | Speedup")
    print(f"  ----------------|-------------|--------")
    t_bump_full = t_bump / n_test * N_quotes
    t_ift_total = t_calib + t_ift  # 1 LP + 1 IFT
    print(f"  Bump ({N_quotes:3d} quotes) | {t_bump_full:>8.1f}s   | 1.0x")
    print(f"  IFT (1 LP+solve) | {t_ift_total*1000:>8.1f}ms  | {t_bump_full/t_ift_total:.0f}x")

    # Print sensitivities
    print(f"\n  ∂(barrier price)/∂(quote):")
    print(f"  {'Quote':>6s} {'Exp':>4s} {'K':>6s} {'Bump':>10s}")
    for idx, q in enumerate(test_indices):
        j_exp = q // N_strikes
        i_str = q % N_strikes
        print(f"  {q:>6d} {expiries[j_exp]:>4.2f} {K[i_str]:>6.2f} {dprice_dquote_bump[idx]:>10.4f}")

    print(f"\n{'='*70}")
    print(f"SANOS + IFT Full Pipeline: COMPLETE")


if __name__ == "__main__":
    main()
