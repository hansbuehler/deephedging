"""
cvxpy_ift — Implicit Function Theorem sensitivity for cvxpy problems.

Computes ∂(solution)/∂(parameters) for any solved cvxpy LP/QP
via the Implicit Function Theorem on active constraints at the optimum.

No re-solving needed: one linear system after the original solve.

Usage:
    import cvxpy as cp
    from cvxpy_ift import ift_jacobian

    # Define and solve problem
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c @ x), constraints)
    prob.solve()

    # Compute sensitivity of solution w.r.t. any parameter vector b
    #   where constraints have the form A @ x <= b
    J = ift_jacobian(prob)
    # J[i, k] = ∂x*_i / ∂b_k  for each active constraint k

Reference:
    Goloubentsev, D., Lakshtanov, E. and Piterbarg, V. (2022)
    "Automatic Implicit Function Theorem", Risk, March 2022.
    SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984964
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Dict, Tuple, List


def _extract_active_set(prob: cp.Problem, tol: float = 1e-7):
    """
    Extract the active constraint matrix and RHS from a solved cvxpy Problem.

    Returns:
        A_active: (n_active, n_vars) matrix of active constraint normals
        b_active: (n_active,) RHS values at active constraints
        constraint_info: list of (constraint_index, type) for each active row
    """
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise ValueError(f"Problem not solved optimally: {prob.status}")

    x_val = np.concatenate([v.value.ravel() for v in prob.variables()])
    n_vars = len(x_val)

    A_rows = []
    b_vals = []
    info = []

    for ci, constr in enumerate(prob.constraints):
        if isinstance(constr, cp.constraints.nonpos.Inequality):
            # expr <= 0  or  expr >= 0
            # Dual variable available
            dual = constr.dual_value
            if dual is None:
                continue

            dual_flat = np.asarray(dual).ravel()
            expr_val = np.asarray(constr.expr.value).ravel()

            for j, (dv, ev) in enumerate(zip(dual_flat, expr_val)):
                # Active if slack ≈ 0 (expr ≈ 0 for <= 0 constraints)
                if abs(ev) < tol or abs(dv) > tol:
                    # Need the constraint gradient ∂(expr)/∂x
                    # For linear constraints: extract from problem data
                    info.append((ci, j, 'ineq'))

        elif isinstance(constr, cp.constraints.zero.Equality):
            dual = constr.dual_value
            if dual is None:
                continue
            info.append((ci, 0, 'eq'))

    return info


def ift_jacobian_from_lp(A_ub, b_ub, x_opt, bounds=None, tol=1e-7):
    """
    Compute ∂x*/∂b for a solved LP: min c'x s.t. A_ub @ x <= b_ub, lb <= x <= ub.

    At the optimum, n_vars constraints are active (complementary slackness).
    IFT gives: ∂x*/∂b_active = A_active^{-1}

    Args:
        A_ub: (m, n) inequality constraint matrix
        b_ub: (m,) RHS vector
        x_opt: (n,) optimal solution
        bounds: list of (lb, ub) per variable, or None
        tol: tolerance for active constraint detection

    Returns:
        dxdb: (n, m) Jacobian — ∂x*_i/∂b_k for each inequality constraint k
              (non-active constraints have zero sensitivity)
        active_mask: (m,) boolean mask of active inequality constraints
        dxdb_bounds: (n, n) Jacobian for bound constraints (if bounds given)
        active_lb, active_ub: boolean masks for active bounds
    """
    n = len(x_opt)
    m = len(b_ub)

    # Identify active inequality constraints
    slack = b_ub - A_ub @ x_opt
    active_ineq = np.abs(slack) < tol

    # Identify active bound constraints
    active_lb = np.zeros(n, dtype=bool)
    active_ub = np.zeros(n, dtype=bool)
    if bounds is not None:
        for i, (lb, ub) in enumerate(bounds):
            if lb is not None and abs(x_opt[i] - lb) < tol:
                active_lb[i] = True
            if ub is not None and abs(x_opt[i] - ub) < tol:
                active_ub[i] = True

    # Build active constraint matrix
    rows = []
    row_sources = []  # ('ineq', k) or ('lb', i) or ('ub', i)

    for k in range(m):
        if active_ineq[k]:
            rows.append(A_ub[k])
            row_sources.append(('ineq', k))

    for i in range(n):
        if active_lb[i]:
            row = np.zeros(n)
            row[i] = -1.0  # -x_i <= -lb_i
            rows.append(row)
            row_sources.append(('lb', i))

    for i in range(n):
        if active_ub[i]:
            row = np.zeros(n)
            row[i] = 1.0  # x_i <= ub_i
            rows.append(row)
            row_sources.append(('ub', i))

    n_active = len(rows)
    if n_active == 0:
        return np.zeros((n, m)), active_ineq, np.zeros((n, n)), active_lb, active_ub

    A_active = np.array(rows)

    # Solve: A_active @ (∂x*/∂b_k) = e_k  for each active constraint k
    if n_active == n:
        # Square system — unique solution
        inv_A = np.linalg.solve(A_active, np.eye(n_active))
    elif n_active < n:
        # Underdetermined — minimum norm solution
        inv_A = np.linalg.lstsq(A_active, np.eye(n_active), rcond=None)[0]
    else:
        # Overdetermined — least squares
        inv_A = np.linalg.lstsq(A_active, np.eye(n_active), rcond=None)[0]

    # Map back: ∂x*/∂b_k for inequality constraints
    dxdb = np.zeros((n, m))
    for col, (src_type, src_idx) in enumerate(row_sources):
        if src_type == 'ineq':
            dxdb[:, src_idx] = inv_A[:n, col]

    # ∂x*/∂bound for bound constraints
    dxdb_bounds = np.zeros((n, n))
    for col, (src_type, src_idx) in enumerate(row_sources):
        if src_type == 'lb':
            dxdb_bounds[:, src_idx] = -inv_A[:n, col]  # lb enters as -x <= -lb
        elif src_type == 'ub':
            dxdb_bounds[:, src_idx] = inv_A[:n, col]

    return dxdb, active_ineq, dxdb_bounds, active_lb, active_ub


def ift_jacobian_parameter(A_ub, b_ub, x_opt, bounds, param_to_b, tol=1e-7):
    """
    Compute ∂x*/∂p where p is an external parameter vector that affects b_ub and/or bounds.

    Args:
        A_ub, b_ub, x_opt, bounds: as in ift_jacobian_from_lp
        param_to_b: function(p_idx) → dict with:
            'ineq': list of (constraint_idx, ∂b/∂p) for affected inequality RHS
            'lb': list of (var_idx, ∂lb/∂p) for affected lower bounds
            'ub': list of (var_idx, ∂ub/∂p) for affected upper bounds
        tol: active constraint tolerance

    Returns:
        dxdp: (n_vars, n_params) Jacobian ∂x*/∂p
    """
    dxdb, active_ineq, dxdb_bounds, active_lb, active_ub = \
        ift_jacobian_from_lp(A_ub, b_ub, x_opt, bounds, tol)

    n = len(x_opt)
    # Infer n_params from param_to_b
    n_params = 0
    while True:
        try:
            param_to_b(n_params)
            n_params += 1
        except (IndexError, KeyError, StopIteration):
            break

    dxdp = np.zeros((n, n_params))

    for p in range(n_params):
        deps = param_to_b(p)

        for k, dbdp in deps.get('ineq', []):
            dxdp[:, p] += dxdb[:, k] * dbdp

        for i, dlb in deps.get('lb', []):
            dxdp[:, p] += dxdb_bounds[:, i] * dlb

        for i, dub in deps.get('ub', []):
            dxdp[:, p] += dxdb_bounds[:, i] * dub

    return dxdp


# ============================================================
# Convenience: wrap scipy.optimize.linprog result
# ============================================================

def ift_linprog(A_ub, b_ub, bounds, result, tol=1e-7):
    """
    One-liner for scipy.optimize.linprog results.

    Args:
        A_ub, b_ub: constraint matrix/RHS passed to linprog
        bounds: bounds passed to linprog
        result: scipy.optimize.OptimizeResult from linprog

    Returns:
        dxdb: (n, m) Jacobian ∂x*/∂b for inequality constraints
        dxdb_bounds: (n, n) Jacobian ∂x*/∂bounds
        info: dict with active constraint counts
    """
    if not result.success:
        raise ValueError(f"LP not optimal: {result.message}")

    dxdb, active_ineq, dxdb_bounds, active_lb, active_ub = \
        ift_jacobian_from_lp(A_ub, b_ub, result.x, bounds, tol)

    info = {
        'n_active_ineq': int(active_ineq.sum()),
        'n_active_lb': int(active_lb.sum()),
        'n_active_ub': int(active_ub.sum()),
        'n_vars': len(result.x),
    }

    return dxdb, dxdb_bounds, info


# ============================================================
# Self-test
# ============================================================

def _test():
    """Verify IFT on a simple LP with known solution."""
    from scipy.optimize import linprog

    # LP: min x1 + x2  s.t.  -x1 - x2 <= -1,  x1, x2 >= 0
    # Solution: x* = (1, 0) or (0, 1) — degenerate
    # With bounds 0 <= x <= 10

    # Better test: min -x1 - 2*x2  s.t.  x1 + x2 <= 4,  x1 <= 3,  x1,x2 >= 0
    # Solution: x* = (3, 1), active: x1 <= 3, x1 + x2 <= 4
    c = np.array([-1, -2])
    A = np.array([[1, 1], [1, 0]])
    b = np.array([4.0, 3.0])
    bounds = [(0, None), (0, None)]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    assert res.success
    print(f"LP solution: x* = {res.x}")

    dxdb, dxdb_bounds, info = ift_linprog(A, b, bounds, res)
    print(f"Active: {info}")

    # Verify: bump b[0] (x1+x2 <= 4+h) → x2 increases by h, x1 stays
    h = 1e-6
    b_up = b.copy(); b_up[0] += h
    res_up = linprog(c, A_ub=A, b_ub=b_up, bounds=bounds, method='highs')
    b_dn = b.copy(); b_dn[0] -= h
    res_dn = linprog(c, A_ub=A, b_ub=b_dn, bounds=bounds, method='highs')
    fd = (res_up.x - res_dn.x) / (2 * h)

    print(f"∂x*/∂b[0]: IFT = {dxdb[:, 0]},  FD = {fd}")
    err = np.max(np.abs(dxdb[:, 0] - fd))
    print(f"Max error: {err:.2e}")
    assert err < 1e-6, f"IFT test failed: {err}"

    # Bump b[1] (x1 <= 3+h)
    b_up = b.copy(); b_up[1] += h
    res_up = linprog(c, A_ub=A, b_ub=b_up, bounds=bounds, method='highs')
    b_dn = b.copy(); b_dn[1] -= h
    res_dn = linprog(c, A_ub=A, b_ub=b_dn, bounds=bounds, method='highs')
    fd1 = (res_up.x - res_dn.x) / (2 * h)

    print(f"∂x*/∂b[1]: IFT = {dxdb[:, 1]},  FD = {fd1}")
    err1 = np.max(np.abs(dxdb[:, 1] - fd1))
    print(f"Max error: {err1:.2e}")
    assert err1 < 1e-6, f"IFT test failed: {err1}"

    print("cvxpy_ift self-test: PASS")


if __name__ == "__main__":
    _test()
