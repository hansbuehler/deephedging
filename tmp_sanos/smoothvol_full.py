"""
Smooth full state dependent model
"""

from dataclasses import dataclass
from cdxcore.util import Timer
from .smoothvol import verify_inp, _bs_call, ExpiryFit, vec_max_dx, bs_implied
from .bs import nb_bs_call
import cvxpy as cp
import numpy as np
import warnings as warnings
import math as math
import gc as gc
from collections.abc import Mapping
from numba import njit, prange
import os as os 
import psutil as psutil

def get_memory_usage_mb():
    """ Function to get process memory usage """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return math.ceil( mem_info.rss / 1024 / 1024 )

def it_bs_kernel( strikes, BKx, BdVx, Bqx, prod_K, sum_dV, prod_q, fac ):
    m    = len(BKx)
    kx   = BKx[0]
    dVx  = BdVx[0]
    BKx  = BKx[1:]
    BdVx = BdVx[1:]
    
    if m == 1:
        ky = prod_K*kx[None,:]
        return nb_bs_call( strikes[:,None] / ky, fac*(sum_dV+dVx[None,:]) ) * ky * prod_q
    
    nr   = len(kx)
    o    = prod_K*0.
    qx   = Bqx[0]
    Bqx  = Bqx[1:]
    for r in range(nr):
        o += it_bs_kernel( strikes, BKx, BdVx, Bqx, prod_K*kx[r], sum_dV+dVx[r], prod_q*qx[r], fac=fac )        
    return o

def make_bs_kernel( strikes, BK, BV, Bq, fac ):
    """
    BS( strikes / prod(BK), sum(BV) ) * prod(BK) * prod(Bq)
    """
    
    m = len(BK)
    assert len(BV) == m
    assert len(Bq) == m-1

    K = it_bs_kernel( strikes=strikes, BKx=BK, BdVx=BV, Bqx=Bq, prod_K=1., sum_dV=0., prod_q=1., fac=fac )
    assert K.shape[0] == len(strikes), ("Shape error", K.shape[0], len(strikes))
    assert K.shape[1] == len(BK[-1]), ("Shape error", K.shape[1], len(BK[-1]))
    return K    

@dataclass
class SmoothComplexCallSurfaceConfig:
    # how to extract modle vols
    dvar_floor     : float = 0.01      # variance must increase that much 
    vol_fac        : float = 0.25      # how to scale model vols
    floor_vol      : float = 0.01      # minimum volatility as floor
    model_strikes  : np.ndarray = None # specify model strikes

    # how to handle bid/ask
    bid_ask_mode   : str = "ignore"    # how to handle bid/ask: 'penalty', 'ignore', 'constraint'
    spread_weighted: bool = True       # weight objective by spread

    # optimizer settings
    solver         : str = "CLARABEL"  # cvxpy solver
    max_iweight    : float = 100.      # maximum input weight, e.g. cap on 1/spread
    max_dx         : float|None = None # maximum strike dx
    min_dx         : float = 1E-6      # minimum strike dx 
    L1             : bool = True       # whether to use L1 for the fit objective vs mid, or L2
    p_smoothness   : float = 0.        # if non-zero add a penalty term p_smoothness*mean(q^2) to the objective.
    mart_theo      : bool = False      # whether to impose no-arb on the jump martingale (theoretically correct) or the actual process (makes more sense)

    dtype          : type = None       # dtype

    @property
    def is_linear(self) -> bool:
        return self.vol_fac is None or self.vol_fac == 0.

class SmoothComplexCallSurface:
    """
    Full complex model fitter.
    ** Experimental. Read the code and use it directly. This code plots prgress until it dies. **
    """
    Config = SmoothComplexCallSurfaceConfig
    def __init__( self, market,
                        config         : SmoothComplexCallSurfaceConfig = SmoothComplexCallSurfaceConfig(),
                        raise_on_error : bool = True,
                        plot_fits      : bool = True,
                        eps            : float = 1E-8):
        """
        Iteratively train model
        """
        M        = len(market)
        verify_inp( config.bid_ask_mode in ['ignore', 'penalty'], lambda : f"'bid_ask' must be one of 'ignore' or 'penalty' not '{config.bid_ask_mode}'")
        verify_inp( M > 0, "'market' cannot be empty")
        market   = list(market.values()) if isinstance(market, Mapping) else market
        dtype    = market[0].dtype if config.dtype is None else config.dtype

        if not config.model_strikes is None:
            verify_inp( len(config.model_strikes.shape) == 1 and len(config.model_strikes) >= 3, "'model_strikes' must be a vector with at least 3 entries")
            verify_inp( config.model_strikes[0] > 0. and config.model_strikes[0] < 1. and config.model_strikes[-1] > 1. and np.min( config.model_strikes[1:] - config.model_strikes[:-1] )>0., "'model_strikes' must be a positive increasing vector, which surrounds 1")

        # Prep
        # ----
        # find extrapolation strikes (heuristic - may need to improve)
    
        atm_vars   = np.zeros( (M,), dtype=dtype )
        min_k      = dtype(1.)
        max_k      = dtype(1.)
        atm_k      = np.ones((1,),dtype=dtype )
        results    = []
        jrvar_r_m1 = 0.
        f0         = dtype(0.)
        for j in range(M):
            # read input & check
            # ------------------
            jdata       = market[j]
            atm_vars[j] = jdata.interpolate( atm_k, "bid_var")[0]
            min_k       = min( min_k, jdata.strikes[0] )
            max_k       = max( max_k, jdata.strikes[-1] )
            if j>0:
                if atm_vars[j] <= atm_vars[j-1]:
                    err = f"Linearly interpolated ATM bid variance is not increasing from expiry {market[j-1].label} {atm_vars[j-1]:.4g} to {jdata.label} {atm_vars[j]:.4g}"
                    if raise_on_error:
                        raise ValueError(err + " -- use 'raise_on_error=False' to turn this error into a warning")
                    warnings.warn(err + " -- ignored, but can cause infeasibility of the LP problem")
                jdata_m1   = market[j-1]
                ixk        = ( jdata.strikes >= jdata_m1.strikes[0] ) & ( jdata.strikes <= jdata_m1.strikes[-1] )
                jprev_asks = market[j-1].interpolate( jdata.strikes[ixk], "asks" )
                err        = np.min( jdata.bids[ixk] - jprev_asks )
                if err<-eps:
                    err = f"Linearly interpolated ask prices from expiry {market[j-1].label} exceed bid prices from expiry {jdata.label}; error {err:.4g}"
                    if raise_on_error:
                        raise ValueError(err + " -- use 'raise_on_error=False' to turn this error into a warning")
                    warnings.warn(err + " -- ignored, but can cause infeasibility of the LP problem")
        min_k = min( 0.1, min_k*0.1 )
        max_k = max( 1.5, max_k*1.5 )

        # prepare output
        # --------------
        for j in range(M):
            jdata          = market[j]
            jr             = ExpiryFit(jdata)
            jr_m1          = results[j-1] if j>0 else None
            assert min_k < jr.strikes[0] and max_k > jr.strikes[-1], ("Internal consistency error?", j, ":", min_k, "<", jr.strikes[0], jdata.strikes[0], "//", max_k, ">", jr.strikes[-1], jdata.strikes[-1] )

            if config.model_strikes is None:
                jr.xstrikes,\
                jr.mx_ix    = vec_max_dx( jr.strikes, min_k=min_k, max_k=max_k, max_dx=config.max_dx, min_dx=config.min_dx )
            else:
                jr.xstrikes = config.model_strikes
                jr.mx_ix    = None
            # compute forward variance per strike
            # note that we set xvar to zero for our boundary strikes;
            # this is not a necessary condition
            jrvar_r_m1     = f0 if j==0 else np.interp( jr.xstrikes[1:-1], jr_m1.xstrikes[1:-1], jrvar_r_m1 ).astype(dtype,copy=False)
            jxvar_r        = jr.interpolate( jr.xstrikes[1:-1], "bid_var")
            if np.min( jxvar_r - jrvar_r_m1 )<0.:
                warnings.warn(f"Expiry {jr.label}: interpolated variance not increasing; error {np.min( jxvar_r - jrvar_r_m1 ):.4g} -- will floor it.")
            jT             = jr.sqrtT**2
            jT_m1          = f0 if j==0 else jr_m1.sqrtT**2
            verify_inp( jT > jT_m1, lambda : f"Expiry {jr.label}: time to expiry is not increasing: found {jT:.4g} and {jT_m1:.4g}")
            jr.xdV         = np.zeros_like( jr.xstrikes )
            jr.xdV[1:-1]   = np.maximum( jxvar_r - jrvar_r_m1, dtype((jT-jT_m1)*config.dvar_floor)) * (config.vol_fac**2)
            jrvar_r_m1     = jxvar_r

            results.append( jr )
        del jr, jT, jT_m1, jdata
        del jrvar_r_m1
        del market

        # Solve iteratively
        # -----------------

        BK         = [ np.ones((1,),dtype=dtype) ]
        BV         = [ np.zeros((1,),dtype=dtype) ]
        Bq         = [ np.ones((1,),dtype=dtype) ]
        jxfloor_m1 = _bs_call( results[0].xstrikes, config.floor_vol * results[0].sqrtT )

        tme_gc = Timer()
        print("Global garbage collection... ", end='', flush=True)
        gc.collect()
        print(f"done; this took {tme_gc}.")
        init_mem = get_memory_usage_mb()

        for j in range(M):
            jr        = results[j]
            jxN       = jr.N

            print(f"\rProcessing expiry {j+1}/{M} {jr.label} with {len(jr.strikes)} market strikes from {jr.strikes[0]:.4g} to {jr.strikes[-1]:.4g} (using {jxN} model strikes): generating tensors...", end='', flush=True)

            tme_kernel = Timer()
            # tensors
            # each B* has shape [jN, ..., 1N]
            # BK = K_j * ... * K_1
            # BV = dV_j * .... * dV_1
            BK.append( jr.xstrikes )
            BV.append( jr.xdV )

            KC          = make_bs_kernel( jr.strikes, BK, BV, Bq, fac=1. )       # kernel to map to model prices on market strikes
            KU          = make_bs_kernel( jr.xstrikes, BK, BV, Bq, fac=0. )        # kernel to map to X prices on model strikes
            tme_kernel  = str(tme_kernel)
            print(f"\rProcessing expiry {j+1}/{M} {jr.label} with {len(jr.strikes)} market strikes from {jr.strikes[0]:.4g} to {jr.strikes[-1]:.4g} (using {jxN} model strikes): produced kernel; this took {tme_kernel}. Now solving linear program... ", end='', flush=True)

            # cvxpy
            # -----
            tme_solve = Timer()
            assert jxfloor_m1.shape == jr.xstrikes.shape, ("Internal error", jxfloor_m1.shape, jr.xstrikes.shape )
            
            jxq         = cp.Variable(jxN)
            jC          = KC @ jxq
            jxU         = KU @ jxq
            jr.xfloor   = jxfloor_m1
            del KC, KU
            assert jC.shape == jr.strikes.shape

            # jq is a unit density
            jconstraints = [ jxq >= 0.,
                             cp.sum(jxq) == 1.,
                             jxq @ jr.xstrikes == 1.,
                             jxU >= jxfloor_m1
                           ]

            # weighting
            weight = None
            if config.spread_weighted:
                weight = np.minimum( dtype(config.max_iweight), 1./jr.spreads )
                assert np.all(np.isfinite(weight)) and np.min(weight) > 0., ("'weight' not positive and finite", j, jr.label,"min weight", np.min(weight))
                weight /= np.sum( weight )

            # fitting
            mid_err = cp.abs( jC-jr.mids )
            mid_err = mid_err if config.L1 else mid_err**2
            mid_err = cp.multiply( weight, mid_err ) if not weight is None else mid_err        
            if config.bid_ask_mode == "penalty":
                ask_err = cp.maximum( 0., jC-jr.asks )
                bid_err = cp.maximum( 0., jr.bids-jC )
                ask_err = cp.multiply( weight, ask_err ) if not weight is None else ask_err
                bid_err = cp.multiply( weight, bid_err ) if not weight is None else bid_err
                err = 0.01 * cp.sum( mid_err ) + cp.sum( ask_err ) + cp.sum( bid_err )
                del ask_err, bid_err, mid_err
            else:
                err = cp.sum( mid_err )
                del mid_err
                
            # smoothness?
            if config.p_smoothness > 0.:
                err += config.p_smoothness * cp.sum_squares(jxq) / float(jxN)

            # solve
            # -----
            prob = cp.Problem( cp.Minimize(cp.sum( err )), constraints=jconstraints )
            try:
                prob.solve(solver=config.solver)# for SCS, eps_abs=1e-8))
            except cp.SolverError as e:
                raise cp.SolverError( f"Expiry {jr.label}: {str(e)}") from e
                
            if prob.status != "optimal":
                status = "" if not config.is_linear else " (linear mode)"
                if prob.status != 'optimal_inaccurate':
                    raise RuntimeError(f"Expiry {jr.label}: solver {config.solver} failed for expiry {jr.label}: {prob.status}{status}")
                warnings.warn(f"Expiry {jr.label}: solver {config.solver} failed for expiry {jr.label}: {prob.status}{status}")
            del prob, jconstraints, err
            
            jr.xdensity  = np.array( jxq.value, dtype=dtype )
            jr.fits      = np.array( jC.value, dtype=dtype )
            jr.Xcalls    = np.array( jxU.value, dtype=dtype )
            del jxq, jC, jxU

            jr.iv_fits   = bs_implied( jr.strikes, jr.fits, sqrtT=jr.sqrtT, warn_only=True, ret_only_vols=True ).astype(dtype,copy=False)
            jr.iv_Xcalls = bs_implied( jr.xstrikes, jr.Xcalls, sqrtT=jr.sqrtT, warn_only=True, ret_only_vols=True ).astype(dtype,copy=False)
            
            tme_solver = str(tme_solve)
            
            # clean up
            # --------

            tme_clean   = Timer()
            print(f"\rProcessing expiry {j} DTE {jr.label} with {len(jr.strikes)} market strikes from {jr.strikes[0]:.4g} to {jr.strikes[-1]:.4g} (using {jxN} model strikes): solved linear program; this took {tme_solver}. Now cleaning up...", end='', flush=True)
            KR          = make_bs_kernel( results[j+1].xstrikes, BK, BV, Bq, fac=0. ) if j<M-1 else None
            jxfloor_m1  = KR @ jr.xdensity if not KR is None else None
            Bq.append( jr.xdensity )
            del KR
            gc.collect()
            tme_clean   = str(tme_clean)
            
            print(f"\rExpiry {j+1}/{M} {jr.label} with {len(jr.strikes)} market strikes from {jr.strikes[0]:.4g} to {jr.strikes[-1]:.4g} (using {jxN} model strikes) done - timing: kernel {tme_kernel} solver {tme_solver} clean {tme_clean}. Memory: {get_memory_usage_mb()-init_mem}MB so far.", flush=True)

            def plot():
                from cdxcore.dynaplot import figure
                # plot
                # ----
                with figure(jr.label, fig_size=(20,5), columns=None) as fig:
                    axC, axV = fig.add_subplots("Calls", "Vols")
    
                    axC.plot( jr.strikes, jr.asks, "v:", color="red", label="ask")
                    axC.plot( jr.strikes, jr.bids, "^:", color="green", label="ask")
                    axC.plot( jr.strikes, jr.fits, "-", color="blue", label="model")
                    
                    axV.plot( jr.strikes, jr.iv_asks, "v:", color="red", label="ask")
                    axV.plot( jr.strikes, jr.iv_bids, "^:", color="green", label="ask")
                    axV.plot( jr.strikes, jr.iv_fits, "-", color="blue", label="model")
    
                    axC.legend(loc="upper right")
                    axV.legend(loc="upper right")

            if plot_fits:
                plot()
            
        self.results = results
        self.config = config
