# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:25:04 2025 by HB
Last Updated on Tues Oct 7 10:26 2025 by RS
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from cdxcore.err import verify, Callable, verify_inp
import warnings as warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from numba import njit, prange
import math as math

from nbs.utils.cdxcore2.verbose import Context

DFLike = Union[pl.DataFrame, pd.DataFrame]

inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
eps_sqrtVar = 1e-9 * ((0.1 / 255.0) ** 0.5)
eps_strike = 1e-9

def bs_call(k: np.ndarray, vol: np.ndarray | float, sqrtT: np.ndarray | float = 1.0):
    r"""
    Compute Black Scholes call option prices in pure price domain.

    The function blends into the intrinsic functions for $$k\downarrow0$ or $\mathrm{vol}\,\sqrt{t} \downarrow0$.

    Parameters
    ----------
    k : np.ndarray
        Strikes.

    vol : np.ndarray | float
        Volatilities or a single volatility.

    sqrtT : np.ndarray | float, default ``1``
        Square-root of time.

    Returns
    -------
    prices : np.ndarray
        Black Scholes call prices.
    """
    verify_inp( np.min(k) >= 0.0, "'k' cannot be negative")
    verify_inp( np.min(vol) >= 0.0, "'vol' cannot be negative")
    verify_inp( np.min(sqrtT) >= 0.0, "'sqrtT' cannot be negative")
    assert np.all(np.isfinite(k + vol + sqrtT)), "Infinite values found"
    vf = vol * sqrtT
    d1 = -np.log(k + eps_strike) / (vf + eps_sqrtVar) + 0.5 * vf
    d2 = d1 - vf
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    C = N1 - k * N2
    C = C
    if isinstance(C, np.ndarray):
        C[vf == 0.0] = np.maximum(1.0 - k[vf == 0.0], 0.0)
        C[k == 0.0] = 1.0
    else:
        if vf == 0.0 or k == 0.0:
            C = np.maximum(1.0 - k, 0.0)
    assert np.all(np.isfinite(C)), "Infinite C"
    return C

@njit(parallel=True, fastmath=True)
def nb_bs_call(k: np.ndarray, sqrtVar: np.ndarray) -> np.ndarray:
    r"""
    Black Scholes numba jit optimized call price function which can be compiled
    within outher numba functions.

    Parameters
    ----------
        k : np.ndarray (n,)
            A numpy vector.
            
        sqrtVar : np.ndarray (n,)
            A numpy vector of the same size as ``k``.
    
    Returns
    -------
        prices : np.ndarray
            A numpy vector of the same size as ``k``.
    """
    calls    = np.empty_like(k)
    invsqrt2 = 1./math.sqrt(2.)

    for i in prange(k.shape[0]):
        for j in prange(k.shape[1]):
            ki = k[i,j]
            sv = sqrtVar[0,j]
            # handle intrinsic cases
            if ki <= 0.0 or sv <= 0.0:
                x = 1.0 - ki
                calls[i,j] = x if x > 0.0 else 0.
            else:
                # regular BS
                d1 = -math.log(ki) / sv + 0.5 * sv
                d2 = d1 - sv
                N1 = 0.5 * (1.0 + math.erf(d1 * invsqrt2))  
                N2 = 0.5 * (1.0 + math.erf(d2 * invsqrt2))  
                calls[i,j] = N1 - ki * N2
    return calls

def bs_price(
    k: np.ndarray,
    vol: np.ndarray | float,
    sqrtT: np.ndarray | float = 1.0,
    is_call: np.ndarray | bool = True,
):
    r"""
    Compute Black Scholes option prices in pure price domain.

    The function blends into the intrinsic functions for $$k\downarrow0$ or $\mathrm{vol}\,\sqrt{t} \downarrow0$.

    Parameters
    ----------
    k : np.ndarray
        Strikes.

    vol : np.ndarray | float
        Volatilities or a single volatility.

    sqrtT : np.ndarray | float, default ``1``
        Square-root of time.

    is_call : np.ndarray | bool, default ``True``
        Whether to compute call (``True``) or put (``False``) prices.

    Returns
    -------
    prices : np.ndarray
        Black Scholes prices.
    """
    C = bs_call(k=k, vol=vol, sqrtT=sqrtT)
    return np.where(is_call, C, C + k - 1.0)  # C-P=1-K


def bs_vega(k: np.ndarray, vol: np.ndarray | float, sqrtT: np.ndarray | float = 1.0):
    r"""
    Compute Black Scholes vega in pure price domain.

    The function blends into zero for $$k\downarrow0$ or $\mathrm{vol}\,\sqrt{t} \downarrow0$.

    Parameters
    ----------
    k : np.ndarray
        Strikes.

    vol : np.ndarray | float
        Volatilities or a single volatility.

    sqrtT : np.ndarray | float, default ``1``
        Square-root of time.

    Returns
    -------
    vega : np.ndarray
        Black Scholes vega.
    """
    assert np.all(np.isfinite(k + vol + sqrtT)), "Infinite values found"
    if np.min(k) <= 0.0:
        raise ValueError("'k' cannot be negative")
    if np.min(vol) < 0.0:
        raise ValueError("'vol' cannot be negative")
    if np.min(sqrtT) < 0.0:
        raise ValueError("'vol' cannot be negative")
    vf = vol * sqrtT
    d1 = (-np.log(k) / vf) + 0.5 * vf
    vega = inv_sqrt_2pi * np.exp(-0.5 * d1**2) * sqrtT

    if isinstance(vega, np.ndarray):
        vega[(vf == 0.0) | (k == 0.0)] = 0.0
        assert np.all(np.isfinite(vega)), "Infinite vega"
    else:
        if vf == 0.0 or k == 0.0:
            vega = 0.0
    return vega

def bs_dk(
    k: np.ndarray,
    vol: np.ndarray | float,
    sqrtT: np.ndarray | float = 1.0,
    is_call: np.ndarray | bool = True,
):
    r"""
    Compute Black Scholes derivative in K

     Parameters
    ----------
    k : np.ndarray
        Strikes.

    vol : np.ndarray | float
        Volatilities or a single volatility.

    sqrtT : np.ndarray | float, default ``1``
        Square-root of time.

    Returns
    -------
    dk : np.ndarray
        Black Scholes derivative in strike.
    """
    assert np.all(np.isfinite(k + vol + sqrtT)), "Infinite values found"
    if np.min(k) < 0.0:
        raise ValueError("'k' most be positive")
    if np.min(vol) <= 0.0:
        raise ValueError("'vol' most be positive")
    if np.min(sqrtT) <= 0.0:
        raise ValueError("'vol' most be positive")

    vf = vol * sqrtT
    d2 = (-np.log(k) / vf) - 0.5 * vf
    N2 = norm.cdf(d2)
    DK = -N2
    if isinstance(DK, np.ndarray):
        DK[vf == 0.0] = np.where(k[vf == 0.0] < 1.0, -1, 0.0)
        DK[k == 0.0] = -1.0
    else:
        if vf == 0.0:
            DK = -1.0 if k < 1.0 else 0.0
        elif k == 0.0:
            DK = -1.0
    assert np.all(np.isfinite(DK)), "Infinite DK"
    DK = np.where(is_call, DK, DK + 1.0)
    return DK

@dataclass
class ErrorStats:
    max_iters: int
    iters: int
    max_err: float
    l1_err: float
    l2_err: float

def bs_implied(
    strikes: np.ndarray,
    prices: np.ndarray,
    is_call: np.ndarray | bool = True,
    sqrtT: np.ndarray | float = 1.0,
    price_tol: np.ndarray | float = 1e-6,
    vol_min: float = 0.01,
    vol_max: float = 5.0,
    max_iters: int = 100,
    eps: float = 1e-10,
    min_vega: float = 1e-12,
    ret_only_vols: bool = True,
    warn_only : bool = False,
    verbose: Context = Context.quiet,
) -> tuple[np.ndarray, np.ndarray, ErrorStats]|np.ndarray:
    """
    Solve for implied volatilities.

    Parameters
    ----------
        strikes : np.ndarray
            Strikes. Must either be of shape ``(N,)`` or of shape ``(N,M)``.
            In the latter case the routine will solve the problem for the surface jointly,

        prices : np.ndarray
            Prices in the same shape as ``strikes``.

        is_call : np.ndarray | bool, default ``True``
            Boolean or boolean array of shape ``(N,)`` indicating
            calls (``True``) or puts (``False``).

        sqrtT : np.ndarray | float, default ``1``
            Square-root of time as float or array of shape ``(N,)``.

        price_tol : np.ndarray | float, default ``1E-6``
            Price tolerance (typically a fraction of spreads) as
            float or array of shape ``(N,)``.

        vol_min : float, default ``0.01``
            Minimum volatility.

        vol_max : float, default ``5``
            Maximum volatility.

        max_iters : int, default ``100``
            Maximum iterations. Usually the routine uses very few iterations.

        min_vega : float, default ``1E-12``
            Minimum vega for taking an updates step.

        ret_only_vols : bool, default ``True``
            Return only vols.

        warn_only : bool, default ``False``
            If ``True``, warn of any input data issues but proceed afterwards. Does not affect consistency errors.

        verbose : :class:`cdxcore.verbose.Context`, default :attr:`cdxcore.verbose.Context.quiet`
            For printing progress information.

    Returns
    -------
        ``vols`` if ``ret_only_vols`` else ``vols``, ``fits``, ``err_stats`` : np.ndarray, np.ndarray, type
            ``vols`` contains the fitted volatilities; ``fits`` contains the black & scholes prices at those values.
            ``err_stats`` is a data class with

              * ``max_iters``: the input ``max_iters``.
              * ``iters``: iterations used.
              * ``max_err``: maximum price error.
              * ``l1_err``: average price error.
              * ``l2_err``: quadratic average price error.
    """
    if np.min(price_tol) < 1e-8:
        raise ValueError(
            f"'price_tol' must be bigger than 1E-8. Found a minimum of {np.min(price_tol):.6g}"
        )
    if strikes.shape[0] != prices.shape[0]:
        raise ValueError(
            f"'strikes' and 'prices' must have the same first dimension; found {strikes.shape} and {prices.shape}"
        )
    if np.min(strikes) < -0.0:
        raise ValueError("'strikes' must be positive")
    if np.min(prices) < 0.0 or np.max(prices) >= 1.0:
        err = f"'prices' must be within [0,1); found min {np.min(prices):.4g} and max {np.max(prices):.4g}"
        if not warn_only:
            raise ValueError(err)
        warnings.warn(err)
        prices = np.maximum( 0., np.minimum( prices, 1. ) )

    shape = prices.shape
    fits = np.zeros_like(prices)  # output prices
    vols = np.zeros_like(prices)  # output vol
    assert np.all(np.isfinite(prices)), "Infinite input prices"

    # convert into same shape
    is_call = (strikes < 0.0) | is_call
    sqrtT = strikes * 0.0 + sqrtT
    price_tol = strikes * 0.0 + price_tol

    if len(prices.shape) == 2:
        # reshape to flat arrays
        strikes = (
            np.full(shape, strikes, dtype=strikes.dtype) if strikes.shape != shape else strikes
        )
        is_call = np.full(shape, is_call, dtype=np.bool_) if is_call.shape != shape else is_call
        sqrtT = np.full(shape, sqrtT, dtype=sqrtT.dtype) if sqrtT.shape != shape else sqrtT
        price_tol = (
            np.full(shape, price_tol, dtype=sqrtT.dtype) if price_tol.shape != shape else price_tol
        )

        prices = prices.reshape((-1,))
        strikes = strikes.reshape((-1,))
        is_call = is_call.reshape((-1,))
        sqrtT = sqrtT.reshape((-1,))
        price_tol = price_tol.reshape((-1,))
        fits = fits.reshape((-1,))
        vols = vols.reshape((-1,))

    assert prices.shape == strikes.shape
    assert prices.shape == is_call.shape
    assert prices.shape == sqrtT.shape
    assert prices.shape == strikes.shape
    assert prices.shape == price_tol.shape
    assert prices.shape == fits.shape
    assert prices.shape == vols.shape

    price_min = bs_price(k=strikes, sqrtT=sqrtT, vol=vol_min, is_call=is_call)
    price_max = bs_price(k=strikes, sqrtT=sqrtT, vol=vol_max, is_call=is_call)
    total = len(fits)

    # boundary cases
    # --------------
    with verbose.write_t(f"Computing implied vols for {total} options:") as tme:
        # identify intrinsic
        intr = np.where(
            is_call,
            np.maximum(0.0, 1.0 - strikes),  # calls
            np.maximum(0.0, strikes - 1.0),
        )  # puts
        # intr           = np.maximum(0., cp * (strikes-1.) )
        done = (strikes <= eps) | (sqrtT <= eps) | (np.abs(intr - prices) <= price_tol)
        fits[done] = intr[done]
        vols[done] = 0.0
        work = ~done
        verbose.report(1, lambda: f"Used intrinsic value for {sum(done)} options.")

        # identify options priced at or beyond vol_max
        done = (price_max - prices <= price_tol) & work
        fits[done] = price_max[done]
        vols[done] = vol_max
        work &= ~done
        verbose.report(1, lambda: f"Used maximum volatility {vol_max:.3f} for {sum(done)} options.")

        # identify options priced at or beyond vol_min
        ## RS FIX: vol_min wasn't masked with work
        # done           = prices - price_min <= price_tol
        done = (prices - price_min <= price_tol) & work
        fits[done] = price_min[done]
        vols[done] = vol_min
        work &= ~done
        verbose.report(1, lambda: f"Used minimum volatility {vol_min:.3f} for {sum(done)} options.")

        # initial guess
        # -------------

        # https://repub.eur.nl/pub/1472/ERS%202004%20054%20FA.pdf (20)
        ## RS MOD: Generalizing to both calls and puts
        S = 1.0
        X = strikes[work]
        C_raw = prices[work]
        isc_work = is_call[work]

        # C-P=1-K => P=C-1+K
        C_eq = np.where(isc_work, C_raw, C_raw - S + X)
        C_eq = np.minimum(np.maximum(C_eq, price_min[work]), price_max[work])  # guard
        vols[work] = (
            np.sqrt(2.0 * np.pi)
            / (2.0 * (S + X))
            * (
                2.0 * C_eq
                + X
                - S
                + np.sqrt(
                    np.maximum(
                        0, (2.0 * C_eq + X - S) ** 2 - 2.0 * (S + X) * ((X - S) ** 2) / (S * np.pi)
                    )
                )
            )
            / sqrtT[work]
        )

        if not np.all(np.isfinite(vols)):
            verbose.report(
                1, lambda: f"**** Found {sum(~np.isfinite(vols))} infinite initial guess vols"
            )
            vols[~np.isfinite(vols)] = vol_max

        vols[work] = np.clip(vols[work], vol_min, vol_max)
        vols_min = np.full_like(vols, vol_min)
        vols_max = np.full_like(vols, vol_max)

        assert np.all(np.isfinite(vols)), "Infinite vols"

        # train
        # -----

        iters = 0
        if sum(work) == 0:
            verbose.report(1, lambda: "*** No further implied volatilities to be computed ***")

        else:
            for iters in range(1, max_iters + 1):
                fits[work] = bs_price(
                    k=strikes[work], sqrtT=sqrtT[work], vol=vols[work], is_call=is_call[work]
                )

                # did we hit it
                done = (np.abs(fits - prices) < price_tol) & work
                work &= ~done

                if sum(work) == 0:
                    verbose.report(
                        2,
                        lambda: f"\rStep {iters}/{max_iters}: implied vol for the remaining {sum(done)} options computed.",
                        end="",
                    )
                    break
                res_max_err = np.max(np.abs(fits[work] - prices[work]) / price_tol[work])
                if tme.interval_test(0.5):
                    verbose.report(
                        2,
                        lambda: f"\rStep {iters}/{max_iters}: implied vol for {sum(done)} options computed; {sum(work)} options with a maximum error of {res_max_err:.4g} left to do.",
                        end="",
                    )
                del done, res_max_err

                # adjust min/max
                too_low = (fits < prices) & work
                vols_min[too_low] = vols[too_low]
                price_min[too_low] = fits[too_low]  # we are not really using these
                too_high = (fits > prices) & work
                vols_max[too_high] = vols[too_high]
                price_max[too_high] = fits[too_high]  # we are not really using these

                # first order:
                # prices(x) = prices(x0) + prices'(x0) * (x-x0)
                # =>
                # x ~ ( prices(x) - prices(x0) ) / prices'(x0) + x0

                test_vega = bs_vega( k=strikes[work], sqrtT=sqrtT[work], vol=vols[work]  )
                new_vols = vols[work] + (prices[work] - fits[work]) / np.maximum( test_vega, min_vega )

                err1 = (vols[work] == vols_min[work]) & (new_vols < vols[work])
                err2 = (vols[work] == vols_max[work]) & (new_vols > vols[work])

                if sum(err1 | err2) > 0:
                    err1 = (
                        f"{sum(err1)} cases where a vega step at the current lower vol would take vol even lower"
                        if sum(err1) > 0
                        else None
                    )
                    err2 = (
                        f"{sum(err2)} cases where a vega step at the current lower vol would take vol even lower"
                        if sum(err2) > 0
                        else None
                    )
                    if err1 is not None and err2 is not None:
                        raise RuntimeError(f"Convergence error: had {err1} and {err2}.")
                    raise RuntimeError(
                        f"Convergence error: had {err1 if err1 is not None else err2}."
                    )

                vols[work] = np.clip(new_vols, vols_min[work], vols_max[work])
                del new_vols

        fits = fits.reshape(shape)
        vols = vols.reshape(shape)
        prices = prices.reshape(shape)

        err_stats = ErrorStats(
            iters=int(iters),
            max_iters=int(max_iters),
            max_err=float(np.max(np.abs(fits - prices))),
            l1_err=float(np.mean(np.abs(fits - prices))),
            l2_err=float(np.std(fits - prices)),
        )

        verbose.write(
            lambda: f"\rImplied vol calculation for {total} options finished using {err_stats.iters}/{err_stats.max_iters} iterations. "
            + f"Max error {err_stats.max_err:.3f}, L1 error {err_stats.l1_err:.3f}, L2 error {err_stats.l2_err:.3f}. "
            + f"This took {tme}. "
        )

    if ret_only_vols:
        return vols
    return vols, fits, err_stats


# make arb free
def mk_arbfree(
    strikes: np.ndarray,
    calls: np.ndarray,
    *,
    spreads: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    p_min: float | np.ndarray = 0,
    l1: float = 0.01,
):
    r"""

    #RS [called as an inner helper from `np_load` excluslively]

    Acts as a **procrustean bed**: it forces the raw market data to conform to convexity *before* training starts. It finds the closest set of prices to the market mids that satisfy:
        $$ C_{i-1} - 2C_i + C_{i+1} \ge 0 $$ (Discrete Convexity)

    `1/vega` used as weights because options with **High Vega** (At-the-money) are liquid and "trusted" (so we penalize moving them), while options with **Low Vega** (tails) are noisy, so the algorithm is allowed to move them more to satisfy convexity constraints.
    ---

    Finds L1 or L2 closest arbitrage-free call prices.

    The function solves for ``C`` such that ``C`` is arbitrage-free
    and minimizes::

        (1-l1)*(err**2) + l1*|err|

    for ``err = (C-calls)*weight``.

    Parameters
    ----------
    strikes : np.ndarray
        Strikes, must be sorted

    calls : np.ndarray
        Input call prices.

    spreads : np.ndarray | None, default ``None``
        Bid/ask spreads. If provided the function will find
        arbitrage-free prices within the implied bids and asks.

        The function returns ``None`` if this fails.

    weight: np.ndarray | None, default ``None``
        Weight to multiply the fitting error with, typically 1./vegaSqrtVar.
        Use ``None`` for unit weights.

    p_min: np.ndarray | float, default ``0``
        Minimum probability mass for each strike.
        This can be a flat number, or an array.
        Since the probability of the last strike is a free variable, only the
        first n-1 entries of ``p_min`` are used in this case.

    l1: float, default ``0.01``
        See above.

    Returns
    -------
    Calls : np.ndarray | ``None``
        Arbitrage-free call prices, or ``None`` if none could be found.
        This can only happen if ``spread`` was provided.
    """
    if len(strikes.shape) != 1 or len(strikes) < 3:
        raise ValueError("'strikes' must be a vector with a length of at least 3")
    
    if strikes.shape != calls.shape:
        raise ValueError(
            f"'strikes' and 'calls' must have the same shape; found {strikes.shape} and {calls.shape}."
        )
    if np.min(strikes[1:] - strikes[:-1]) <= 0.0:
        raise ValueError("'strikes' must be sorted and strictly increasing")
    if np.min(strikes) <= 0.0:
        raise ValueError(f"Lowest strike must exceed 0; found {np.min(strikes):.4g}.")
    if l1 < 0.0 or l1 > 1.0:
        raise ValueError("'l1' must be within [0,1]")

    if spreads is not None:
        if strikes.shape != spreads.shape:
            raise ValueError(
                f"'strikes' and 'spreads' must have the same shape; found {strikes.shape} and {spreads.shape}."
            )
        if np.min(spreads) <= 0.0:
            ValueError("'spreads' must be positive")
    if weights is not None:
        if strikes.shape != weights.shape:
            raise ValueError(
                f"'strikes' and 'weights' must have the same shape; found {strikes.shape} and {weights.shape}."
            )
        if np.min(weights) < 0.0:
            ValueError("'weights' must not be negative")
    if isinstance(p_min, np.ndarray):
        if p_min.shape[0] == strikes.shape[0]:
            p_min = p_min[:-1]
        else:
            if not p_min.shape[0] == strikes.shape[0]:
                raise ValueError("'p_min' must have as many entries as strikes, or one less.")
    else:
        p_min = np.full((strikes.shape[0] - 1,) + strikes.shape[1:], p_min, dtype=strikes.dtype)
    assert np.min(p_min) >= 0.0, ("'p_min' cannot be negative", np.min(p_min))
    p_min = np.maximum(p_min, 1e-5)

    extra_strike = max(strikes[-1] * 2, strikes[-1] + (strikes[-1] - strikes[-2]) * 4)
    I   = np.maximum(1.0 - strikes, 0.0)
    C   = cp.Variable(len(calls))
    dC  = (C[1:] - C[:-1]) / (strikes[1:] - strikes[:-1])  # +1 = P[X<=k[:-1]]
    dC0 = (C[0] - 1.0) / (strikes[0] - 0.0)
    dCL = (0.0 - C[-1]) / (extra_strike - strikes[-1])
    constraints = [
        p_min[0] <= dC[0] - dC0,
        p_min[1:] <= dC[1:] - dC[:-1],
        0. <= dCL - dC[-1],  #=
    ]

    if spreads is not None:
        bids = calls - spreads / 2.0
        asks = calls + spreads / 2.0
        constraints += [C >= bids, C <= asks]

    err = cp.multiply(weights, C - calls) if weights is not None else (calls - C)
    if l1 == 0.0:
        err = err**2
    elif l1 == 1.0:
        err = cp.abs(err)
    else:
        err = (1 - l1) * (err**2) + l1 * cp.abs(err)

    prob = cp.Problem(cp.Minimize(cp.sum(err)), constraints=constraints)
    prob.solve(solver="CLARABEL")# for SCS, eps_abs=1e-8)
    if prob.status != "optimal":
        prob.solve(solver="SCS", eps_abs=1e-8)

    if prob.status != "optimal":
        assert spreads is not None, (
            "Internal error: LP program should not fail if 'spreads' was not provided"
        )
        return None
    C  = np.array(C.value).astype(calls.dtype, copy=False)
    C  = np.maximum( C, I )
    dC = np.diff(C, prepend=1.0) / np.diff(strikes, prepend=0.0)
    assert np.min(dC) >= -1.0, (
        "Internal error - arbitrage is violated: dC must exceed -1",
        np.min(dC),
    )
    assert np.min(dC[1:] - dC[:-1]) >= -1E-8, (
        "Internal error - arbitrage is violated: dC not increasing",
        np.min(dC[1:] - dC[:-1]),
        dC,
    )
    return C


def mk_smooth_arbfree(
    strikes: np.ndarray,
    calls: np.ndarray,
    *,
    spreads: np.ndarray,
    weights: np.ndarray | None = None,
    call_weight: float = 1.0,
):
    """
    Finds call prices within bid/ask which are smooth in the sense that
    the L2 norm of the second derivatives is minimized.
    Adds strikes between market strikes to do so.
    """
    if len(strikes.shape) != 1:
        raise ValueError("'strikes' must be a vector")
    if strikes.shape != calls.shape:
        raise ValueError(
            f"'strikes' and 'calls' must have the same shape; found {strikes.shape} and {calls.shape}."
        )
    if np.min(strikes[1:] - strikes[:-1]) <= 0.0:
        raise ValueError("'strikes' must be sorted and strictly increasing")
    if np.min(strikes) <= 0.0:
        raise ValueError(f"Lowest strike must exceed 0; found {np.min(strikes):.4g}.")
    if strikes.shape != spreads.shape:
        raise ValueError(
            f"'strikes' and 'spreads' must have the same shape; found {strikes.shape} and {spreads.shape}."
        )
    if np.min(spreads) <= 0.0:
        ValueError("'spreads' must be positive")
    if strikes.shape != weights.shape:
        raise ValueError(
            f"'strikes' and 'weights' must have the same shape; found {strikes.shape} and {weights.shape}."
        )
    if np.min(weights) < 0.0:
        ValueError("'weights' must not be negative")

    first_new = max(strikes[0] / 2.0, strikes[0] - (strikes[1] - strikes[0]))
    last_new = strikes[-1] + 10 * (strikes[-1] - strikes[-2])
    assert first_new > 0.0 and first_new < strikes[0]
    assert last_new > strikes[-1]

    new_strikes = np.zeros(((strikes.shape[0] - 1) * 2 + 1 + 2,), dtype=strikes.dtype)
    new_strikes[0] = first_new
    new_strikes[-1] = last_new
    new_strikes[1:-1:2] = strikes
    new_strikes[2:-2:2] = (strikes[1:] + strikes[:-1]) / 2.0
    assert np.min(new_strikes[1:] - new_strikes[:-1]) > 0.0, "Internal strike error"
    ext_strike = new_strikes[-1] + (new_strikes[-1] - new_strikes[-2]) * 4.0
    bids = calls - spreads / 2.0
    asks = calls + spreads / 2.0

    I = np.maximum(1.0 - new_strikes, 0.0)
    C = cp.Variable(len(new_strikes))
    dC = (C[1:] - C[:-1]) / (new_strikes[1:] - new_strikes[:-1])  # +1 = P[X<=k[:-1]]
    dC0 = (C[0] - 1.0) / (new_strikes[0] - 0.0)
    dCL = (0.0 - C[-1]) / (ext_strike - new_strikes[-1])
    new_bids = np.zeros(((bids.shape[0] - 1) * 2 + 1,), dtype=bids.dtype)
    new_asks = np.zeros_like(new_bids)
    new_bids[0::2] = bids
    new_asks[0::2] = asks
    new_bids[1:-1:2] = (bids[1:] + bids[:-1]) / 2.0
    new_asks[1:-1:2] = (asks[1:] + asks[:-1]) / 2.0
    constraints = [
        C >= I,
        0.0 <= dC[0] - dC0,
        0.0 <= dC[1:] - dC[:-1],
        0.0 <= dCL - dC[-1],
        C[1:-1] >= new_bids,
        C[1:-1] <= new_asks,
    ]

    def ferr(x):
        return x**2

    err = (
        cp.sum((dC[1:] - dC[:-1]) ** 2)
        + (dC0 - dC[1]) ** 2
        + ferr(dCL - dC[-1]) ** 2
        + call_weight * cp.sum((cp.multiply(weights, C[1:-1:2] - calls)) ** 2)
    )

    prob = cp.Problem(cp.Minimize(err), constraints=constraints)
    prob.solve()
    if prob.status != "optimal":
        assert spreads is not None, (
            "Internal error: LP program should not fail if 'spreads' was not provided"
        )
        return None
    C = np.array(C.value).astype(calls.dtype, copy=False)
    dC = np.diff(C, prepend=1.0) / np.diff(new_strikes, prepend=0.0)
    assert np.min(dC) >= -1.0, (
        "Internal error - arbitrage is violated: dC must exceed -1",
        np.min(dC),
    )
    assert np.min(dC[1:] - dC[:-1]) >= 0.0, (
        "Internal error - arbitrage is violated: dC not increasibg",
        np.min(dC[1:] - dC[:-1]),
        dC,
    )
    return C, new_strikes, C[1:-1:-2]


def remove_eps_strikes(strikes : np.ndarray, *args, eps : float=0.0001) -> tuple:
    """
    Sorts the input strikes and removes any strikes which are closer than ``eps``.
    """
    if len(strikes.shape) != 1:
        raise ValueError("'strikes' must be a vector")
    ixs = np.argsort(strikes)
    strikes = strikes[ixs]
    if strikes[-1] - strikes[0] <= eps:
        raise ValueError(
            "The difference between the biggest and smallest strike is less than 'eps'"
        )
    strikes_ = [strikes[0]]
    outs = [[a[0]] for a in args]
    for i in range(1, len(strikes)):
        if strikes[i] - strikes_[-1] > eps:
            strikes_.append(strikes[i])
            for o, a in zip(outs, args):
                o.append(a[i])
    strikes = np.array(strikes_, dtype=strikes.dtype)
    outs = [np.array(o, dtype=strikes.dtype) for o in outs]
    return tuple([strikes] + outs)
