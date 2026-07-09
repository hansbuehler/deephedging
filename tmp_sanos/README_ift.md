# IFT + AAD Sensitivity for SANOS LP Calibration

## What this does

Computes **exact sensitivities** of any downstream exotic price
with respect to **all vanilla option quotes** — through the LP calibration.

- **Part I (Python):** IFT through LP → ∂(fitted surface)/∂(quotes). No re-calibration, one linear solve.
- **Part II (C++/AADC):** AAD through MC pricing → ∂(exotic price)/∂(surface). One reverse pass.
- **Chain rule:** ∂(exotic)/∂(quotes) = ∂(exotic)/∂(surface) × ∂(surface)/∂(quotes).

## The problem

You calibrated a SANOS surface from N vanilla quotes. Now you price an exotic (barrier, autocallable) and need:

```
∂(exotic price) / ∂(vanilla quote_i)    for all i = 1, ..., N
```

**Standard approach**: bump each quote, re-calibrate LP, re-price exotic. Cost: 2N × (LP + MC).

**This approach**: IFT through LP + AAD through MC. Cost: 1 LP + 1 linear solve + 1 MC (forward + reverse).

## Quick start

### Part I: IFT sensitivity (pure Python, no AADC needed)

```python
from example_sanos_ift import calibrate_sanos, sanos_sensitivity

# Your market data
K = np.linspace(0.75, 1.25, 15)
mids = market_mid_prices(K)
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

### Part II: End-to-end with AADC

```python
# Step 1-2: Same as above (IFT Jacobian)
J_ift = sanos_sensitivity(A_ub, b_ub, bounds, lp_result, len(K))

# Step 3: Downstream pricing with AADC
#   C++ program records: surface → local vol → MC barrier → price
#   One reverse pass → ∂price/∂(all surface nodes)
result = subprocess.run(['./sanos_downstream_aadc', surface_file], ...)
price, dV_dsurface = parse_output(result.stdout)

# Step 4: Chain rule
dV_dquotes = dV_dsurface @ J_ift
# All N sensitivities from 1 LP + 1 linear solve + 1 AAD MC
```

## How it works

### IFT through LP (Part I)

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

### AAD downstream (Part II)

The MC pricing kernel is recorded on an AADC tape:
- Inputs: surface nodes σ²(K_m, T_j)
- Local vol interpolation (on tape, differentiable)
- GBM path simulation
- Barrier/payoff evaluation
- Output: discounted payoff

One forward + one reverse pass gives ∂price/∂σ²(K_m) for ALL grid nodes simultaneously.

### Chain rule

```
∂V/∂quote_i = Σ_m (∂V/∂σ²_m) · (∂σ²_m/∂C_fit_m) · (∂C_fit_m/∂quote_i)
                   ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
                   AAD (C++)     Dupire (analytic)   IFT (Python)
```

## Files

| File | Description |
|------|-------------|
| `example_sanos_ift.py` | Self-contained: calibrate + IFT + validate vs bump + benchmark |
| `cvxpy_ift.py` | Reusable IFT module for any LP (scipy.linprog) |
| `sanos_full_test.py` | Multi-expiry LP + IFT + downstream MC (pure Python) |
| `sanos_e2e_test.py` | End-to-end: Python IFT + C++ AAD + chain rule |
| `sanos_downstream_aadc.cpp` | C++/AADC: barrier MC with reverse-mode AD |

## Benchmarks

| Setup | Bump time | IFT+AAD time | Speedup |
|-------|-----------|-------------|---------|
| 15 quotes, IFT only | 38 ms | 1.6 ms | **24×** |
| 33 quotes, IFT only | 340 ms | 8.5 ms | **40×** |
| 33 quotes, end-to-end (LP+MC) | 602 s | 9.1 s | **66×** |

## Dependencies

- **Part I (Python)**: numpy, scipy (linprog with HiGHS solver)
- **Part II (C++/AADC)**: [AADC library](https://matlogica.com/aadc) by MatLogica

## References

1. Buehler, Horvath, Kratsios, Limmer, Saqur (2026). "SANOS: Smooth strictly Arbitrage-free Non-parametric Option Surfaces." [arXiv:2601.11209](https://arxiv.org/abs/2601.11209)
2. Goloubentsev, Lakshtanov, Piterbarg (2022). "Automatic Implicit Function Theorem." Risk, March 2022. [SSRN 3984964](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984964)
3. Dupire (1994). "Pricing with a smile." Risk, 7(1), 18-20.
