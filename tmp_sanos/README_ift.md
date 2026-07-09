# IFT Sensitivity for SANOS LP Calibration

## What this does

Computes **exact sensitivities** of a SANOS-calibrated surface
with respect to **all vanilla option quotes** — without re-calibrating.

No finite differences through the LP. One linear system solve.

## The problem

You calibrated a SANOS surface from N vanilla quotes. Now you need:

```
∂C_fit(K_i) / ∂quote_j    for all i, j = 1, ..., N
```

**Standard approach**: bump each quote, re-calibrate LP, re-price. Cost: 2N LP solves.

**This approach**: IFT through LP. Cost: 1 LP + 1 linear solve.

## Quick start

```python
from example_sanos_ift import calibrate_sanos, sanos_sensitivity

# Your market data
K = np.linspace(0.75, 1.25, 15)      # strikes (normalized, forward=1)
mids = market_mid_prices(K)           # mid quotes
bids = mids - spread; asks = mids + spread

# Step 1: Calibrate (LP — fits within bid-ask, enforces butterfly ≥ 0)
C_fit, lp_result, A_ub, b_ub, bounds = calibrate_sanos(K, mids, bids, asks)

# Step 2: Full Jacobian — ONE call, no re-solving
J = sanos_sensitivity(A_ub, b_ub, bounds, lp_result, len(K))

# J[i,j] = ∂C_fit(K_i) / ∂mid(K_j)
# J[i,i] = 1.0  if quote passes through (no binding arb constraint)
# J[i,i] = 0.0  if quote is pinned by butterfly/calendar constraint
# J[i,j] ≠ 0    cross-sensitivity through arbitrage constraints
```

## How it works

At the LP optimum, n constraints are active (slack = 0):

```
A_active @ x* = b_active
```

By the Implicit Function Theorem:

```
∂x*/∂b = A_active⁻¹
```

The active set is returned by the LP solver. One linear system solve gives the
full Jacobian. Exact (not approximate) because LP active constraints are exact
equalities.

This is an application of the Automatic IFT framework of
[Goloubentsev, Lakshtanov, Piterbarg (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984964)
to the LP case, where the IFT is exact (vs approximate for least-squares).

## Files

| File | Description |
|------|-------------|
| `example_sanos_ift.py` | Self-contained example: calibrate + IFT + validate vs bump |
| `cvxpy_ift.py` | Reusable IFT module for any LP (scipy.linprog) |

## Benchmarks

| Setup | Bump time | IFT time | Speedup |
|-------|-----------|----------|---------|
| 15 quotes | 38 ms | 1.6 ms | **24×** |
| 33 quotes | 340 ms | 8.5 ms | **40×** |

## Dependencies

- numpy, scipy (linprog with HiGHS solver)

## References

1. Buehler, Horvath, Kratsios, Limmer, Saqur (2026). "SANOS: Smooth strictly Arbitrage-free Non-parametric Option Surfaces." [arXiv:2601.11209](https://arxiv.org/abs/2601.11209)
2. Goloubentsev, Lakshtanov, Piterbarg (2022). "Automatic Implicit Function Theorem." Risk, March 2022. [SSRN 3984964](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984964)
