"""
Non-homogeneous strike fitting via the NHMG formulation.

This module prepares market data, formulates the cvxpy linear program, solves it,
and maps the fitted prices back to the input dataframe.
"""

from .bs import bs_implied
from cdxcore.pretty import PrettyObject
from cdxcore.err import verify_inp
from cdxcore.util import fmt_filename
from cdxcore.dynaplot import figure, m_o_m
from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass
import cvxpy as cp
import numpy as np
import warnings as warnings
from scipy.stats import norm
import math as math
from datetime import date
from scipy.interpolate import PchipInterpolator

def focus_line(left : float, 
               right : float, 
               N : int,*,
               concentration : float = 0.9,
               power : float|int=2,
               focus : float|None = None,
               eps : float=1E-8,
               dtype : type = None ):  # TODO: move into cdxcore
    r"""
    Returns a line from ``left`` to ``right`` which has more points in ``focus``.

    This function computes a number of points on a line between ``left`` and ``right`` such that points around ``focus``
    are more dense. The ``concentration`` parameter allowws to blend between linear and focused points.

    The function is an appropriately scaled and shifted version of $x \mapsto x (1-c) + c\, x|x|^{p-1}$ where $c$ is concentration.

    Parameters
    ----------
        left, right : float, float
            Left hand and right hand points.

        N : int,
            Number of points; must be at least 3.

        concentration : float, default ``0.9``
            How much concentration: maximum is ``1`` while ``0`` give a linear distibution of points.

        focus : float|None, default ``None``
            Focus point. If not provided, the mid-point is used.

        dtype : type, default ``None``
            Numpy dtype.

    Returns
    -------
        points : :class:`numpy.ndarray`
            Numpy vector
    """

    left = float(left)
    right = float(right)
    if not focus is None:
        verify_inp( left<focus and focus<right, "'left', 'focus', and 'right' must be in order")
    else:
        verify_inp( left<right, "'left' and 'right' must be in order")
        focus = 0.5 * (left+right)
        
    verify_inp( N>=3, "'N' must be at least 3")
    verify_inp( concentration >=0. and concentration <= 1., "'concentration' must be within [0,1]")

    x     = np.linspace( - 1.,  +1., N )
    x     = x*(1.-concentration)+concentration*x*( np.pow(np.abs(x), power-1 ) if power>1 else 1.)
    assert np.abs( x[0]+x[-1] )<1E-8, ("Internal error", x[0], x[-1] )    
    x     /= x[-1]
    x[x<0.] *= (left-focus)/x[0]
    x[x>0.] *= (right-focus)/x[-1]
    x     += focus
    x[0]  = left
    x[-1] = right 
    return x

def vec_max_dx( strikes : np.ndarray, *, min_k : float, max_k : float, max_dx : float|None = None, min_dx : float = 0. ) -> tuple:
    """
    Generate model strikes.
    
    This function returns a vector of strikes such that:
        
    * Lowest strike is ``min_k``
    * Highest strike is ``max_k``
    * Strikes are at least ``min_dx`` apart from each other. Strikes which violate this are dropped iteratively from the front.
    * Strikes in the original strike range are at most ``max_dx`` apart from each other. If None, do not add strikes except min_k or max_k. Outside the extreme strikes of ``strikes``` 
      the function adds strikes with ``10 max_dx``.
    
    The function returns a tuple, the first element is the new strike vector.
    The second element represents the index positions of the ``strikes`` in the new vector. An index is -1 if an element was dropped due to ``min_k``.
    The function returns ``slice(1,-1)`` if
    only min_k, max_k were added ie the returned strikes are [min_k, strike[0], ..., strike[-1], max_k].    
    """
    verify_inp( min_k > 0. and min_k < 1. and max_k > 1., "'min_k' must be in (0,1) and 'max_k' must be above 1" )
    verify_inp( min_k < strikes[0] and max_k > strikes[-1], "'min_k' and 'max_k' must be outside 'strikes'")
    verify_inp( max_dx is None or max_dx >= 0., "'max_dx' must be None or non-negative")
    verify_inp( min_dx >= 0., "'min_dx' must be non-negative")

    if max_dx is None:
        if np.max( strikes[1:]-strikes[:-1] ) >= min_dx:
            out = np.empty( (len(strikes)+2,), dtype=strikes.dtype )
            out[1:-1] = strikes
            out[0]    = min_k
            out[-1]   = max_k
            return out, slice(1,-1)
        
        out = [ min_k ]
        six = []
        for x in strikes[1:]:
            dx = x - out[-1]
            if dx <= min_dx:
                six.append(-1)
                continue
            else:
                six.append( len(out ))
                out.append( x )
        return np.asarray( out, dtype=strikes.dtype ), np.asarray( six, dtype=np.int32 )        
            
    # max_dx
    # ------

    outer_max_dx = min(0.25,max_dx*10.) 
    inner_max_dx = max_dx

    left    = strikes[0]-inner_max_dx
    out     = np.linspace( min_k, left, max(2,1+math.floor( (left - min_k)/outer_max_dx )),dtype=strikes.dtype).tolist()\
              if left > min_k else\
              [ min_k ]    
    six     = [ len(out) ]
    out.append( strikes[0] )
    
    for x in strikes[1:]:
        dx = x - out[-1]
        if dx <= min_dx:
            six.append(-1)
            continue
        if dx > inner_max_dx:
            n = max(1,int(dx/inner_max_dx))+2
            r = np.linspace( out[-1], x, n )
            for x in r[1:]:
                out.append(x)
            six.append( len(out)-1 )
        else:
            six.append( len(out ))
            out.append( x )

    right = strikes[-1]+inner_max_dx
    rout  = np.linspace( right, max_k, max(2,1+math.floor( (max_k-right)/outer_max_dx )),dtype=strikes.dtype).tolist()\
            if right < max_k else\
            [ max_k ]
   
    out += rout
    out  = np.array( out, dtype=strikes.dtype )
    six  = np.asarray( six, dtype=np.int32 ) if len(out) > len(strikes)+2 else slice(1,-1)
    assert np.all( out[six] == strikes ), ("Internal error", out[six][:20], strikes[:20] )
    return np.asarray( out, dtype=strikes.dtype ), six      

def _bs_call(k: np.ndarray, sqrtVar: np.ndarray ):
    r"""
    Compute Black Scholes call option prices in pure price domain.
    Simplified function for our SmoothCallSurface
    """
    verify_inp( np.min(k) >= 0.0, "'k' cannot be negative")
    verify_inp( np.min(sqrtVar) >= 0.0, "'sqrtVar' cannot be negative")
    assert np.all(np.isfinite(k + sqrtVar)), "Infinite values found"
    
    calls   = np.zeros_like( k )
    inix    = ( (k <= 0.) | (sqrtVar <= 0.))
    calls[inix] = np.maximum( 1.-k, 0. )[inix]
    k       = np.where( inix, 1., k )
    sqrtVar = np.where( inix, 0.1, sqrtVar )
    
    d1 = -np.log(k) / sqrtVar + 0.5 * sqrtVar
    d2 = d1 - sqrtVar
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    calls[~inix] = ( N1 - k * N2 )[~inix]
    assert np.all(np.isfinite(calls)), "Infinite calls"
    return calls

class ExpiryData(PrettyObject):
    r"""
    Market data for a single expiry.

    Market data for ``smoothvol`` are (pure)[https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1141877] market data which means they are scaled by discount factor and forward: if $F_j$ and $\mathrm{DF}_j$ are forward and discount factor
    for expiry $T_j$, and if $c^i_j:=c(T_j,k^i_j)$ are option prices for strikes $k^i_j$ then assuming proportional dividends:

    * $K^i_j := k^i_j/F_j$ are the "pure" strikes and
    * $C^i_J := c^i_j/F_j/\mathrm{DF}_j$ are pure market prices.

    Parameters
    ----------
        label : str|int
            An identifying label for this expiry, for example days to expiry.

        strikes : np.ndarray
            Pure strikes of size (N,)

        bids, asks : np.ndarray
            Bid and ask prices, each of size (N,)

        sqrtT : float
            Sqrt of time-to-expiry in years.
            When converting a real life date to time to expiry it is recommended to use actual business days, for example via
            `pandas-market-calendars <https://pandas-market-calendars.readthedocs.io/en/latest/usage.html>`_.

        iv_bids, iv_asks :  np.ndarray | None, default ``None``
            Bid and ask implied vols, each of size (N,).
            If ``None`` they will be computed using ``bs_implied``.

        dtype : type, default ``np.float32``
            Data type to convert data to.

        eps : float, default ``1E-8``
            Numerical precision for testing data validity.
    """
    def __init__(self,  label    : str|int, 
                        strikes  : np.ndarray,
                        bids     : np.ndarray, 
                        asks     : np.ndarray, 
                        sqrtT    : float,*,
                        iv_bids  : np.ndarray|None = None,
                        iv_asks  : np.ndarray|None = None,
                        dtype    : type = np.float32,
                        eps      : float = 1E-8
                ):
        self.label    = str(label)        
        self.dtype    = dtype
        self.strikes  = np.asarray( strikes, dtype=dtype )
        self.bids     = np.asarray( bids, dtype=dtype )
        self.asks     = np.asarray( asks, dtype=dtype )
        self.sqrtT    = np.asarray( sqrtT, dtype=dtype )
        self.iv_bids  = np.asarray( iv_bids, dtype=dtype ) if not iv_bids is None else None
        self.iv_asks  = np.asarray( iv_asks, dtype=dtype ) if not iv_asks is None else None
        self.intr     = np.maximum( dtype(1.) - self.strikes, dtype(0.) )
        self.spreads  = self.asks - self.bids
        self.mids     = 0.5 * ( self.asks + self.bids )
        
        verify_inp( len(self.strikes.shape) == 1, lambda : f"Expiry {self.label}: 'strikes' must be a vector")
        verify_inp( len(self.strikes) >= 2, lambda : f"Expiry {self.label}: must have at least two 'strikes'")
        verify_inp( np.min( self.strikes[1:] - self.strikes[:-1] ) > 0., lambda : f"Expiry {self.label}: 'strikes' must be increasing.")
        verify_inp( self.strikes[0] > 0., lambda : f"Expiry {self.label}: 'strikes' must be positive")
        verify_inp( self.strikes[0] < 1. and self.strikes[-1] > 1., lambda : f"Expiry {self.label}: 'strikes' must surround 1. Found outer strikes {self.strikes[0]:.4g} and {self.strikes[-1]:.4g}")
        verify_inp( self.sqrtT > 0., "'sqrt' must be positive")

        verify_inp( self.strikes.shape == self.bids.shape, lambda : f"Expiry {self.label}: 'strikes' and 'bids' must have the same shape; found {self.strikes.shape} and {self.bids.shape}")
        verify_inp( self.strikes.shape == self.asks.shape, lambda : f"Expiry {self.label}: 'strikes' and 'asks' must have the same shape; found {self.strikes.shape} and {self.asks.shape}")
        verify_inp( np.min( self.spreads ) > 0., lambda : f"Expiry {self.label}: 'bids' must be strictly below 'asks'; found minimum spread of {np.min( self.spread ):.4g}")   
        verify_inp( np.max( self.asks-1. ) < eps, lambda : f"Expiry {self.label}: 'asks' must not exceed 1. Found {np.max( self.asks ):.4g}")
        verify_inp( np.min( self.bids - self.intr ) >= -eps, lambda : f"Expiry {self.label}: 'bids' must exceed instrinsic; found error {np.min( self.bids - self.intr ):.4g}")

        # floor bids and asks
        self.asks = np.minimum( self.asks, dtype(1.) )
        self.bids = np.maximum( self.bids, self.intr )
        self.mids = np.maximum( self.mids, self.bids+self.spreads*0.1 )
        self.asks = np.maximum( self.asks, self.mids+self.spreads*0.1 )
        
        # implied vols
        if self.iv_bids is None:
            self.iv_bids = bs_implied( self.strikes, self.bids, sqrtT=self.sqrtT, ret_only_vols=True, warn_only=True ).astype(dtype,copy=False)
        else:
            verify_inp( self.iv_bids.shape == self.bids.shape, lambda : f"Expiry {self.label}: 'iv_bids' and 'bids' must have the same shape; found {self.iv_bids.shape} and {self.bids.shape}")

        if self.iv_asks is None:
            self.iv_asks = bs_implied( self.strikes, self.asks, sqrtT=self.sqrtT, ret_only_vols=True, warn_only=True ).astype(dtype,copy=False)
        else:
            verify_inp( self.iv_asks.shape == self.asks.shape, lambda : f"Expiry {self.label}: 'iv_asks' and 'asks' must have the same shape; found {self.iv_asks.shape} and {self.asks.shape}")

    @property
    def T(self) -> float:
        return self.sqrtT**2

    def interpolate(self, strikes: np.ndarray, what: str) -> np.ndarray:
        """
        Interpolate linearly in strike space.

        Parameters
        ----------
            strikes : np.ndarray, float
                Pure strikes to interpolate for.

            what : str
                What to interpolate: ``"mid"``, ``"bids"``, ``"asks" prices, ``"bid_vol"``, ``"ask_vol"``, ``"bid_sqrtVar"``, ``"ask_sqrtVar"``, , ``"bid_var"``, ``"ask_var"``.
                Note that this is naive linear interpolation and does *not* use the SANOS smooth vol function here.

        Returns
        -------
            Result : np.ndarray
        """
        strikes = np.asarray(strikes, dtype=self.dtype)

        match what:
            case "mids" | "bids" | "asks":
                y = getattr(self, what)
            
            case "bid_vol" | "ask_vol":
                y = self.iv_bids if what[0] == 'b' else self.iv_asks

            case "bid_sqrtVar" | "ask_sqrtVar":
                base = self.iv_bids if what[0] == 'b' else self.iv_asks
                y = base * self.sqrtT

            case "bid_var" | "ask_var":
                base = self.iv_bids if what[0] == 'b' else self.iv_asks
                t = base * self.sqrtT
                y = t * t
             
            case _:
                raise ValueError(
                "Unknown interpolation target. Expected one of:"
                "mids, bids, asks, bid_vol, bid_sqrtVar, bid_var, ask_vol, ask_sqrtVar, ask_var. "
                f"Got '{what}'."
            )

        return np.interp(strikes, self.strikes, y).astype(self.dtype, copy=False)

class ExpiryFit( ExpiryData ):
    """
    Fitted SANOS model for one expiry.

    This class is generated by ``SmoothCallSurfaceConfig``.
    The ``__call__`` function interpolates using `SANOS <https://arxiv.org/abs/2601.11209>`_.
    """

    def __init__(self, market : ExpiryData):
        for k, v in market.items():
            self[k] = v

        self.xstrikes       = None
        self.xvols          = None
        self.xfloor         = None  
        self.xdensity       = None
        self.xfits          = None
        self.iv_fits        = None
        self.atm_var        = None
        self.atm_vol        = None
        self.atm_call       = None
        self.mx_ix          = None  # self.strikes == self.xstrikes[self.mx_ix] or None if such mapping does not exist

    @property
    def N(self) -> int:
        """ Number of (extended) strikes just as in `SANOS <https://arxiv.org/abs/2601.11209>`_. """
        return self.xstrikes.shape[0]

    def _assign_result( self, *, jq, eps : float = 1E-8 ):
        """ assign cvxpy results """
        jq     = np.array( jq.value, dtype=self.dtype ) 
        assert jq.shape == self.xstrikes.shape, ("Shape error", jq.shape, self.xstrikes.shape )
        err1   = min( np.min(jq),0.)
        err2   = np.abs( np.sum(jq)-1. ) 
        err3   = np.abs( jq @ self.xstrikes - 1. ) 
        if err1<-eps*100. or err2>eps*100. or err3>eps*100.:
            warnings.warn(f"Expiry {self.label}: output density has issues: "
                          f"positivity (minimum element): {err1:.6g}; "
                          f"probability differnece to 1: {err2:.6g}; "
                          f"expectation difference to 1: {err3:.6g}; eps is {eps:.6g}")
        self.xdensity = np.maximum(0., jq)
        self.xfits,\
        self.iv_xfits = self( self.xstrikes, and_vols=True )
        self.fits,\
        self.iv_fits  = self( self.strikes, and_vols=True )
        self.atm_call,\
        self.atm_vol  = self( 1., and_vols = True )
        assert isinstance(self.atm_call, (np.number, float)), ("???", self.atm_call)
        self.atm_var  = (self.atm_vol*self.sqrtT)**2

    def __call__( self, strikes : np.ndarray|float, *, and_vols : bool = False ) -> np.ndarray|float|tuple[np.ndarray,np.ndarray]|tuple[float,float]:
        """
        Compute call prices for ``strikes`` using the `SANOS <https://arxiv.org/abs/2601.11209>`_ model.

        If ``and_vols`` is provided, also compute implied vols
        """
        if isinstance(strikes, (float, int)):
            r = self( np.array( [strikes], dtype=self.dtype ), and_vols=and_vols )
            if and_vols:
                calls, vols = r
                assert len(calls) == 1 and len(vols) == 1, ("Internal error", calls, vols)
                return calls[0], vols[0]
            assert len(r) == 1 and isinstance(r, np.ndarray), ("Internal error", r)
            return r[0] 

        strikes = np.asarray( strikes, dtype=self.dtype )
        assert len(strikes.shape) == 1, ("Strikes must be a vector", strikes.shape)        
        calls = ( _bs_call( strikes[:,None] / self.xstrikes[None,:], self.sqrtT*self.xvols[None,:] ) * self.xstrikes[None,:] ) @ self.xdensity
        assert calls.shape == strikes.shape
        if not and_vols:
            return calls
        vols = bs_implied( strikes, calls, sqrtT=self.sqrtT, warn_only=True, ret_only_vols=True ).astype(self.dtype,copy=False)
        return calls, vols
    
    def _priceT( self, strikes : np.ndarray|float, *, sqrtT : float|None = None ) -> np.ndarray|float|tuple[np.ndarray,np.ndarray]|tuple[float,float]:
        """
        Compute call prices for ``strikes`` using the `SANOS <https://arxiv.org/abs/2601.11209>`_ model.
        This function allows overwriting ``sqrtT`` for time interpolation.
        """
        if isinstance(strikes, (float, int)):
            r = self( np.array( [strikes], dtype=self.dtype ), and_vols=and_vols )
            return r[0] if not and_vols else (r[0], r[1])

        sqrtT   = sqrtT if not sqrtT is None else self.sqrtT   
        strikes = np.asarray( strikes, dtype=self.dtype )
        calls = ( _bs_call( strikes[:,None] / self.xstrikes[None,:], self.sqrtT*self.xvols[None,:] ) * self.xstrikes[None,:] ) @ self.xdensity
        assert calls.shape == strikes.shape
        return calls
    
    def linear_market( self, strikes : np.ndarray, *, and_vols : bool = False ) -> np.ndarray:
        """
        Compute calls prices for ``strikes`` using linear interpolation from market mid-prices
        (which means they might not be arbitarage free).

        If ``and_vols`` is provided, also compute implied vols
        """
        xmstrikes, xmcalls = self.xm_strikes_calls()
        calls              = np.interp( strikes, xmstrikes, xmcalls ).astype(self.dtype)
        if not and_vols:
            return calls
        return calls, bs_implied( strikes, calls, sqrtT=self.sqrtT, warn_only=True, ret_only_vols=True ).astype(self.dtype,copy=False)
    
    def xm_strikes_calls(self):
        """
        Returns the market extended strike and call range, i.e.
        ``xmstrikes`` are ``strikes`` extended on the boundaries by the boundaries of ``xstrikes``; ``xmcalls`` are
        then extended to be intrinsic value at those strikes.
        """
        xmstrikes     = np.zeros((self.strikes.shape[0]+2,), dtype=self.dtype)
        xmcalls       = np.zeros((self.strikes.shape[0]+2,), dtype=self.dtype)
        xmstrikes[1:-1] = self.strikes
        xmcalls[1:-1] = self.mids
        xmcalls[0]    = 1.-self.xstrikes[0]
        xmcalls[-1]   = 0.
        xmstrikes[0]  = self.xstrikes[0]
        xmstrikes[-1] = self.xstrikes[-1]
        return xmstrikes, xmcalls

    @property
    def stats( self ):
        """
        Returns a dictionary of error fitting stats:

        * ``bid_ask_err``: price excess outside bid/ask, per strike
        * ``bid_ask_rel_err``: price excess as fraction of spread, per strike
        * ``iv_bid_ask_err``: implied vol excess outside bid.ask, per strike
        """
        bid_ask_err     = np.maximum( self.fits - self.asks, 0. ) + np.maximum( self.bids - self.fits, 0. )
        return FitStats( 
            bid_ask_err     = bid_ask_err,
            bid_ask_rel_err = bid_ask_err / self.spreads,
            iv_bid_ask_err  = np.maximum( self.iv_fits - self.iv_asks, 0. ) + np.maximum( self.iv_bids - self.iv_fits, 0. )
            )
        
class FitStats(PrettyObject):
    """
    Fitting stats

    Simple container class which summarizes fitting stats.
    Such a stat is generated by expiry by ``SmoothCallSurface``.
    """
    class ErrStats(PrettyObject):
        def __init__(self, errors : np.ndarray|None = None ):
            self.l1    = 0.   # L1 error
            self.l2sq  = 0.   # L2 error^2
            self.el1   = 0.   # L1 error of all errors
            self.el2sq = 0.   # L2 error^2 of all errors
            self.max   = 0.   # Maximum error
            self.n     = 0    # number of observsations
            self.nerr  = 0    # number of errors
            self += errors
            
        @property
        def l2(self) -> float:
            return math.sqrt( self.l2sq )
        @property
        def el2(self) -> float:
            return math.sqrt( self.el2sq )
        
        def __iadd__(self, errors = None):
            if errors is None:
                return self
            if isinstance(errors, type(self)):
                if errors.n == 0:
                    return self
                nn          = self.n+errors.n
                nnerr       = self.nerr+errors.nerr
                self.l1     = (self.l1*self.n + errors.l1*errors.n) / nn
                self.l2sq   = (self.l2sq*self.n + errors.l2sq*errors.n) / nn
                self.el1    = (self.el1*self.nerr + errors.el1*errors.nerr) / nnerr if errors.nerr > 0 else self.el1
                self.el2sq  = (self.el2sq*self.nerr + errors.el2sq*errors.nerr) / nnerr if errors.nerr > 0 else self.el2sq
                self.max    = max( self.max, errors.max )
                self.n      = nn
                self.nerr   = nnerr
            else:
                if len(errors) == 0.:
                    return self
                assert np.min(errors) >= 0., "'errors' cannot be negative"
                eerrors    = errors[errors>0.]
                nn         = self.n+len(errors)
                nnerr      = self.nerr+len(eerrors)
                self.l1    = ( self.l1*self.n + np.sum(np.abs(errors)) )/nn
                self.l2sq  = ( self.l2sq*self.n + np.sum(errors**2) )/nn
                self.el1   = ( self.el1*self.nerr + np.sum(np.abs(eerrors)) )/nnerr if len(eerrors) > 0 else self.el1
                self.el2sq = ( self.el2sq*self.nerr + np.sum(eerrors) )/nnerr if len(eerrors) > 0 else self.el2
                self.max   = max( self.max, np.max(eerrors) ) if len(eerrors) > 0 else self.max
                self.n     = nn
                self.nerr  = nnerr
            return self
                
    def __init__(self, *, bid_ask_err = None, bid_ask_rel_err = None, iv_bid_ask_err = None):
        self.bid_ask_err     = self.ErrStats()
        self.bid_ask_rel_err = self.ErrStats()
        self.iv_bid_ask_err  = self.ErrStats()        
        self.bid_ask_err     += bid_ask_err
        self.bid_ask_rel_err += bid_ask_rel_err
        self.iv_bid_ask_err  += iv_bid_ask_err
            
    def __iadd__(self, other ):
        assert isinstance(other, type(self)), ("'other' must be of same type",type(self),"not",type(other))
        assert not self.bid_ask_err is None
        self.bid_ask_err     += other.bid_ask_err
        self.bid_ask_rel_err += other.bid_ask_rel_err
        self.iv_bid_ask_err  += other.iv_bid_ask_err
        return self

    def __add__(self, other):
        r = FitStats()
        r += self
        r += other
        return r

@dataclass
class SmoothCallSurfaceConfig:
    """ Configuration for ``SmoothCallSurface``."""
    dtype          : type = None       # dtype. If not specify, infer from first strike array.
    no_warnings    : bool = False      # suppress warnings

    # how to extract model vols
    vol_mode       : str = "atm"       # how to compute model volatilities: 'atm', 'min' or 'strip'. The model is guaranteed to be arbitrage-free for 'atm' and 'min'. Vols will be multiplied by 'vol_fac'
    vol_fac        : float = 0.25      # how to scale model vols
    floor_vol      : float = 0.01      # minimum volatility as floor
    model_strikes  : np.ndarray = None # specify model strikes. Must be an increasing vector of positive strikes surrounding 1

    # how to handle bid/ask
    bid_ask_mode   : str = "ignore"    # how to handle bid/ask: 'penalty', 'ignore', 'constraint'
    spread_weighted: bool = True       # weight objective by spread

    # optimizer settings
    solver         : str = "CLARABEL"  # cvxpy solver
    method         : str = "global"    # "global" fit, "iterative" fit or "individual" fit
    max_iweight    : float = 100.      # maximum input weight, e.g. cap on 1/spread
    max_dx         : float|None = 0.05 # maximum strike dx
    min_dx         : float = 1E-6      # minimum strike dx 
    mart_theo      : bool = False      # whether to impose no-arb on the jump martingale (theoretically correct) or the actual process (makes more sense)

    L1             : bool = True       # (not very useful) whether to use L1 for the fit objective vs mid, or L2
    p_smoothness   : float = 0.        # (not very useful) if non-zero add a penalty term p_smoothness*mean(q^2) to the objective.

    @property
    def is_linear(self) -> bool:
        return self.vol_fac is None or self.vol_fac == 0.

class SmoothCallSurface( object ):
    """
    SANOS model fit to multiple expiries.
    """
    Config = SmoothCallSurfaceConfig
    
    def __init__( self, market         : Mapping[ExpiryData], *,
                        config         : SmoothCallSurfaceConfig = SmoothCallSurfaceConfig(),
                        raise_on_error : bool = True,
                        eps            : float = 1E-8
                ):
        """ fit model to market  """
        M        = len(market)
        verify_inp( config.bid_ask_mode in ['ignore', 'penalty', 'constraint'], lambda : f"'bid_ask' must be one of 'ignore', 'penalty', 'constraint' not '{config.bid_ask_mode}'")
        verify_inp( M > 0, "'market' cannot be empty")
        market   = list(market.values()) if isinstance(market, Mapping) else market
        dtype    = market[0].dtype if config.dtype is None else config.dtype

        if not config.model_strikes is None:
            verify_inp( len(config.model_strikes.shape) == 1 and len(config.model_strikes) >= 3, "'model_strikes' must be a vector with at least 3 entries")
            verify_inp( config.model_strikes[0] > 0. and config.model_strikes[0] < 1. and config.model_strikes[-1] > 1. and np.min( config.model_strikes[1:] - config.model_strikes[:-1] )>0., "'model_strikes' must be a positive increasing vector, which surrounds 1")

        # Prep
        # ----
        # find extrapolation strikes (heuristic - may need to improve)
        atm_vars = np.zeros( (M,), dtype=dtype )
        min_k    = dtype(1.)
        max_k    = dtype(1.)
        atm_k    = np.ones((1,),dtype=dtype )
        for j in range(M):
            # read input & check
            # ------------------
            jdata       = market[j]
            atm_vars[j] = jdata.interpolate( atm_k, "bid_var")
            min_k       = min( min_k, jdata.strikes[0] )
            max_k       = max( max_k, jdata.strikes[-1] )
            if j>0:
                if atm_vars[j] <= atm_vars[j-1]:
                    err = f"Linearly interpolated ATM bid variance is not increasing from expiry {market[j-1].label} {atm_vars[j-1]:.4g} to {jdata.label} {atm_vars[j]:.4g}"
                    if raise_on_error:
                        raise ValueError(err + " -- use 'raise_on_error=False' to turn this error into a warning")
                    if not config.no_warnings:
                        warnings.warn(err + " -- ignored, but can cause infeasibility of the LP problem")
                jdata_m1   = market[j-1]
                ixk        = ( jdata.strikes >= jdata_m1.strikes[0] ) & ( jdata.strikes <= jdata_m1.strikes[-1] )
                jprev_asks = market[j-1].interpolate( jdata.strikes[ixk], "asks" )
                err        = np.min( jdata.bids[ixk] - jprev_asks )
                if err<-eps:
                    err = f"Linearly interpolated ask prices from expiry {market[j-1].label} exceed bid prices from expiry {jdata.label}; error {err:.4g}"
                    if raise_on_error:
                        raise ValueError(err + " -- use 'raise_on_error=False' to turn this error into a warning")
                    if not config.no_warnings:
                        warnings.warn(err + " -- ignored, but can cause infeasibility of the LP problem")
            del jdata
        min_k = min( 0.1, min_k*0.1 )
        max_k = max( 1.5, max_k*1.5 )

        # prepare output
        # --------------
        results  = []
        for j in range(M):
            jdata             = market[j]
            jr                = ExpiryFit(jdata)
            assert min_k < jr.strikes[0] and max_k > jr.strikes[-1], ("Internal consistency error?", j, ":", min_k, "<", jr.strikes[0], jdata.strikes[0], "//", max_k, ">", jr.strikes[-1], jdata.strikes[-1] )
            if config.model_strikes is None:
                jr.xstrikes,\
                jr.mx_ix      = vec_max_dx( jr.strikes, min_k=min_k, max_k=max_k, max_dx=config.max_dx, min_dx=config.min_dx )
            else:
                jr.xstrikes   = config.model_strikes
                jr.mx_ix      = None
            jxN               = len(jr.xstrikes)
            jr.xvols          = np.empty( (jxN,), dtype=jdata.dtype )
            
            # what vols to use for Y.
            if config.is_linear:
                jr.xvols[:]    = 0.
            elif config.vol_mode == "atm":
                jr.xvols[:]    = config.vol_fac * np.sqrt( atm_vars[j] ) / jdata.sqrtT
            elif config.vol_mode == "min":
                jr.xvols[:]    = config.vol_fac * np.min( jdata.iv_bids )
            else:
                verify_inp( config.vol_mode == "market", lambda : f"'vol_mode' must be 'atm', 'min', or 'market', not '{config.vol_mode}'")
                jr.xvols[1:-1] = config.vol_fac * jdata.interpolate( jr.xstrikes[1:-1], "bid_vol" )
                jr.xvols[0]    = jr.xvols[1]
                jr.xvols[-1]   = jr.xvols[-2]
            results.append( jr )
            del jr, jxN, jdata
        del atm_vars, atm_k
 
        # Solve
        # -----
        gerr         = 0. if config.method == "global" else None
        gconstraints = [] if config.method == "global" else None
        jqs          = [] if config.method == "global" else None

        for j in range(M):
            jr    = results[j]
            jxN   = len(jr.xstrikes)
            jr_m1 = results[j-1] if j>0 and config.method != "individual" else None
            
            # core LP
            # -------
            # martingale marginal
            jxq          = cp.Variable( jxN )   
            jconstraints = [ jxq >= 0.,
                             cp.sum( jxq ) == 1.,
                             jxq @ jr.xstrikes == 1.
                           ]

            def jkernel( strikes ):
                return ( _bs_call( strikes[:,None] / jr.xstrikes[None,:], jr.sqrtT*jr.xvols[None,:] ) *  jr.xstrikes[None,:] ) \
                       if not config.is_linear else\
                       np.maximum( jr.xstrikes[None,:] - strikes[:,None], 0. )
                

            # calls at model strikes and market strikes
            jxC          = jkernel( jr.xstrikes ) @ jxq
            jC           = jxC[jr.mx_ix] if not jr.mx_ix is None else ( jkernel( jr.strikes ) @ jxq )
            assert jC.shape == jr.strikes.shape, ("Shape error", jC.shape, jr.strikes.shape )

            # no arbitrage
            if j==0 or config.method == "individual":
                # fit indivudually - will not impose term structure arbitrage
                # impose floor per xstrike
                jr.xfloor     = _bs_call( jr.xstrikes, jr.sqrtT*config.floor_vol ) if not config.is_linear else np.maximum( 1. - jr.xstrikes, 0. )
                jconstraints += [ jxC >= jr.xfloor ]
                jconstraints  += [ jxC >= jr.xfloor ]
                
            elif config.method == "iterative":
                assert not jr_m1 is None, "jr_m1"
                assert not jr is None, "jr"
                assert not jr_m1.xstrikes is None, "jr_m1.xstrikes"
                # local fit
                # impose floor above previous expiry priced at current xstrikes
                jr.xfloor     = jr_m1(jr.xstrikes)
                assert jxC.shape == jr.xfloor.shape, ("Internal error", jxC.shape, jr.xfloor.shape )
                jconstraints += [ jxC >= jr.xfloor ]
                
            else:
                verify_inp( config.method == "global", lambda : f"'method' must be 'global', 'iterative', or 'individual', not '{config.method}']")
                # global fit
                # ensure to exceed the previous calls at previous strikes
                jxq_m1        = jqs[j-1]
                jxC_m1        = ( _bs_call( jr.xstrikes[:,None] / jr_m1.xstrikes[None,:], jr_m1.sqrtT*jr_m1.xvols[None,:] ) *  jr_m1.xstrikes[None,:] ) @ jxq_m1 \
                                if not config.is_linear and not config.mart_theo else\
                                np.maximum( jr_m1.xstrikes[None,:] - jr.xstrikes[:,None], 0. ) @ jxq_m1
                
                jr.xfloor     = None
                jconstraints  += [ jxC >= jxC_m1 ]
                del jxC_m1, jxq_m1
            del jxC
                
            # bid/ask
            # check that current bid exceeds previous option value
            if config.bid_ask_mode == "constraint":
                if j>0 and config.method == "iterative":
                    floor         = jr_m1( jr.strikes )
                    floor_msg     = f"previous expiry {results[j-1].label}"
                elif not config.is_linear:
                    floor         = _bs_call( jr.strikes, jr.sqrtT*config.floor_vol ) 
                    floor_msg     = f"the 'floor_vol' of {config.floor_vol:.4g}"
                else:
                    floor         = np.maximum( 1. - jr.strikes, 0. )
                    floor_msg     = "intrinsic value"

                if np.min( jr.bids - floor ) <-eps:
                    err = f"Expiry {jr.label}: 'bids' are below floor imposed by {floor_msg}"
                    if raise_on_error:
                        raise ValueError(err + " -- use 'raise_on_error=False' to turn this error into a warning")
                    if not config.no_warnings:
                        warnings.warn(err + " -- ignored, but this can cause infeasibility of the LP problem")
                jconstraints += [ jC >= jr.bids, jC <= jr.asks ]  # remain within bid/ask

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
            if config.method == "global":
                gerr         += err
                gconstraints += jconstraints
                jqs.append( jxq )
                
            else:
                prob = cp.Problem( cp.Minimize(cp.sum( err )), constraints=jconstraints )
                try:
                    prob.solve(solver=config.solver)# for SCS, eps_abs=1e-8))
                except cp.SolverError as e:
                    raise cp.SolverError( f"Expiry {jr.label}: {str(e)}") from e
                    
                if prob.status != "optimal":
                    status = f"using 'method={config.method}' and 'bid_ask_mode={config.bid_ask_mode}'" 
                    if config.is_linear:
                        status += " (linear)"
                    if prob.status != 'optimal_inaccurate':
                        raise RuntimeError(f"Expiry {jr.label}: solver {config.solver} failed : {prob.status} {status}")
                    warnings.warn(f"Expiry {jr.label}: solver {config.solver} failed: {prob.status} {status}")
                jr._assign_result( jq=jxq, eps=eps )
                assert not jr.xfits is None

        # global optimization
        # -------------------
    
        if config.method == "global":
            prob = cp.Problem( cp.Minimize(cp.sum( gerr )), constraints=gconstraints )
            try:
                prob.solve(solver=config.solver)# for SCS, eps_abs=1e-8))
            except cp.SolverError as e:
                raise cp.SolverError( f"Global fit: {str(e)}") from e
            if prob.status != "optimal":
                status = f"using 'method={config.method}' and 'bid_ask_mode={config.bid_ask_mode}'" 
                if config.is_linear:
                    status += " (linear)"
                if prob.status != 'optimal_inaccurate':
                    raise RuntimeError(f"Solver {config.solver} failed for global fit: {prob.status} {status}") 
                if not config.no_warnings:
                    warnings.warn(f"Solver {config.solver} returned suboptimal global fit: {prob.status} {status}")
    
            for j, (jr, jxq) in enumerate(zip(results,jqs)):
                jr._assign_result( jq=jxq, eps=eps )          
            
        # find stats
        # ----------
        stats = FitStats()
        for jr in results:
            stats    += jr.stats
            atm_call,\
            atm_vol  =  jr( [1], and_vols = True )
        
        self.results  = results
        self.stats    = stats
        self.config   = config

        # time interpolation
        # ------------------

        if len(results) > 2:
            Ts = np.array( [0.] + [ jr.T for jr in results ], dtype=dtype )
            Vs = np.array( [0.] + [ jr.atm_var for jr in results ], dtype=dtype )
            self.atm_intp = PchipInterpolator( Ts, Vs, extrapolate=True )
            self.expiries = [ ]
        
    @property
    def is_linear(self) -> bool:
        """ Whether this surface represents in fact a linear fit """
        return self.config.is_linear
    
    def __call__( self, strikes : np.ndarray|float, expiry : float):
        """ Call prices for any expiry, with monotonic spline interpolation in ATM variance and extrapolation with terminal vols. """        
        strikes = np.asarray( strikes )
        verify_inp( expiry >= 0., "'expiry' cannot be negative")
        is_float = len(strikes).shape == 0
        strikes  = strikes.reshape((-1,)) if is_float else strikes      
        
        if expiry == 0.:
            return np.maximum(1.-strikes, 0.)

        if expiry >= self.expiries[-1]:
            # extrapolation beyond last expiry: just use the terminal vols
            calls     = self._price( strikes, sqrtT=math.sqrt( expiry )   )
            return calls if not is_float else calls[0]

        atm_var  = self.atm_intp( expiry )
        atm_call = _bs_call( 1., math.sqrt( atm_var ) )

        if expiry <= self.expiries[0]:
            lower_atm_call = 0.
            lower_calls = np.maximum(1.-strikes, 0.)
        else:
            ix = self.expiries.searchsorted(expiry)
            assert ix>0 and ix<self.M, ("Internal error", ix, self.M)  
            lower_atm_call = self.results[ix-1].atm_call
            lower_calls = np.maximum(1.-strikes, 0.)

        upper_atm_call = self.results[ix].atm_call
        upper_calls    = self.results[ix]( strikes )            
        
        assert lower_atm_call < upper_atm_call, ("ATM prices not increasing...?", lower_atm_call, upper_atm_call, lexp, uexp)
        alpha  = ( atm_call - lower_atm_call ) / ( upper_atm_call - lower_atm_call )
        calls  = lower_calls * (1.-alpha) + upper_calls * alpha 
        return calls if not is_float else calls[0]
        
    def plot( self, *, 
              show_CA   : bool = True, 
              show_C    : bool = True, 
              plot_floor_vols: bool = True,
              plot_prev_vols: bool = True,
              vdetails  : PrettyObject|None = None,
              density_x : bool = False,
              fdx      : float = 0.001,
              dt       : str|date|None = None,
              col_size : int = 6, 
              row_size : int = 4,
              out_dir: str | None = None,
              show: bool = True,
              num_focus_strikes:int=201
              ):
        """
        Plots the surface
        """        
        dt = dt.strftime("%Y-%m-%d") if isinstance(dt, date) else dt
        out_path = None if out_dir is None else Path(out_dir)
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
        for j, jr in enumerate(self.results):
            with figure(f"{f'Smooth' if not self.is_linear else 'Linear'} fitted expiry {jr.label}" +\
                        ("" if dt is None else f" @ "+dt)+\
                        ("" if self.is_linear else rf" $\eta={self.config.vol_fac**2:.2f}$"),
                        columns=None, col_size=col_size, row_size=row_size) as fig:
                axCA, axC, axV, axD, axVD   = fig.add_subplots("Calls (extended)" if show_CA else None,
                                                               "Calls" if show_C else None, 
                                                               "Vols", 
                                                               "Density", 
                                                               "Vol Detail" if not vdetails is None else None)
                fig.render(draw=show)

                # to test interpolation generate more strikes 
                # strikes ~ market, xstrikes ~ model strikes
                test_xstrikes         = focus_line( jr.xstrikes[0]*0.9, jr.xstrikes[-1]*1.1, num_focus_strikes,power=10., focus = 0.5*(jr.strikes[0]+jr.strikes[-1]) )
                test_strikes          = focus_line( jr.strikes[0], jr.strikes[-1], num_focus_strikes )
                test_xcalls           = jr(test_xstrikes, and_vols=False)
                test_calls, test_vols = jr(test_strikes, and_vols=True)
                test_xdC  =          np.gradient(test_xcalls, test_xstrikes, edge_order=2)
                test_xq  =           np.gradient(test_xdC,      test_xstrikes, edge_order=2)
                txix                   = ( jr.xstrikes >= jr.strikes[0] ) & ( jr.xstrikes <= jr.strikes[-1] ) # map model strikes to market strike range

                # error bars
                C_spread               = jr.asks - jr.bids
                C_err                  = ( np.maximum( jr.fits - jr.asks, 0.) + np.minimum( jr.fits - jr.bids , 0. ) )/C_spread
                V_spread               = jr.iv_asks - jr.iv_bids
                V_err                  = ( np.maximum( jr.iv_fits - jr.iv_asks, 0.) + np.minimum( jr.iv_fits - jr.iv_bids , 0. ) )/V_spread

                # market
                market_xcalls          = jr.linear_market( test_xstrikes )
                market_calls, market_vols = jr.linear_market( test_strikes, and_vols=True )
                market_xdC  =          np.gradient(market_xcalls, test_xstrikes, edge_order=2)
                market_xp  =           np.gradient(market_xdC,      test_xstrikes, edge_order=2)
                
                # compute density for the market
                iv_xfloor              = bs_implied( jr.xstrikes, jr.xfloor, sqrtT=jr.sqrtT, ret_only_vols=True, warn_only=True ) if not jr.xfloor is None and plot_floor_vols else None
    
                if j>0:
                    jr_prev                = self.results[j-1]
                    prev_calls, prev_vols  = jr_prev( jr.strikes, and_vols = True )
                    prev_vols              *= jr_prev.sqrtT/jr.sqrtT
    
                # plot extended calls
                # -------------------
                
                if not axCA is None:
                    axCA.plot( test_xstrikes, market_xcalls, "*:", color="orange", label="target")
                    axCA.plot( jr.xstrikes, jr.xfits, ".", color="blue", label="fit (at target") 
                    axCA.plot( test_xstrikes, test_xcalls, ":", color="blue", label="fit (interpolated)") 
                    axCA.legend(loc="lower left") 
                    axCA.set_xlabel("Extended strikes")
                    axCA.set_xlim( test_xstrikes[0]-fdx, test_xstrikes[-1]+fdx )
    
                # plot calls
                # ----------

                if not axC is None:
                    axC.plot(jr.strikes,jr.bids, "^", color="green", label="bid")
                    axC.plot(jr.strikes,jr.asks, "v", color="red", label="ask")
                    axC.plot(jr.strikes,jr.fits, ".", color="blue", label="fit (market)")
                    axC.plot(jr.xstrikes[txix],jr.xfits[txix], "*", color="orange", label="fit (extended)")
                    axC.plot(test_strikes,test_calls, ":", color="blue", label="fit (interpolated)")
                    axC.legend(loc="lower left") 
                    axC.set_xlabel("Strikes")
                    axC.set_xlim( jr.strikes[0]-fdx, jr.strikes[-1]+fdx )
    
                    C_err_max = np.abs( np.max(C_err) )
                    if C_err_max>1E-6:
                        axC_2 = axC.twinx()
                        axC_2.plot( jr.strikes, jr.strikes*0., ":", color="blue", alpha=0.2)
                        axC_2.bar( jr.strikes, C_err, width=0.001, color="blue", alpha=0.2, label="Fit error")
                        axC_2.set_ylim( -C_err_max, C_err_max )
                        axC_2.set_ylabel("Fitting error / spread")
                        axC_2.legend(loc="upper right")
    
                # plot vol, vol detail
                # --------------------
    
                def plot_vol( ax, vdetails = None ):
                    if ax is None:
                        return
                    if not vdetails is None:
                        lo_k     = vdetails.get("lo_k", jr.strikes[0])
                        hi_k     = vdetails.get("hi_k", jr.strikes[-1])
                        verify_inp( lo_k < hi_k, lambda : f"'vdetails' lo_k {lo_k:.4g} must be below hi_k {hi_k:4.g}")
                        det_ixs  = ( jr.strikes >= lo_k ) & ( jr.strikes <= hi_k ) 
                        det_xixs = ( jr.xstrikes >= lo_k ) & ( jr.xstrikes <= hi_k ) 
                    else:
                        det_ixs = slice(None)
                    
                    ax.plot(jr.strikes,jr.iv_bids, "^", color="green", label="bid")
                    ax.plot(jr.strikes,jr.iv_asks, "v", color="red", label="ask")
                    ax.fill_between(jr.strikes,jr.iv_bids, jr.iv_asks, color="orange", alpha=0.2, label="bid/ask")
                    ax.plot(jr.strikes,jr.iv_fits, ".", color="blue", label="fit (market)")
                    ax.plot(test_strikes,test_vols, ":", color="blue", label="fit (interpolated)")
                    if iv_xfloor is not None and plot_floor_vols:
                        # plotting the whole strike range here -> need to adjust ylim below
                        ax.plot(jr.xstrikes[txix],iv_xfloor[txix], "--", color="red", label="floor")
                    if j>0 and plot_prev_vols:
                        ax.plot(jr.strikes,prev_vols, "2", color="orange", label="previous")
                    ax.legend(loc="lower left") 
                    ax.set_xlabel("Strikes") 

                    V_err_max = np.max(np.abs(V_err[det_ixs]))
                    if V_err_max>1E-6:
                        ax_2 = ax.twinx()
                        ax_2.plot( jr.strikes, jr.strikes*0., ":", color="blue", alpha=0.2)
                        ax_2.bar( jr.strikes, V_err, width=0.001, color="blue", alpha=0.2, label="Fit error")
                        ax_2.set_ylim( -V_err_max, V_err_max )
                        ax_2.set_ylabel("Vol fitting error / vol spread")
                        ax_2.legend(loc="upper right")
                                            
                    if vdetails is None:
                        ax.set_xlim( jr.strikes[0]-fdx, jr.strikes[-1]+fdx )
                        ax.set_ylim( m_o_m( jr.iv_bids, jr.iv_asks, jr.iv_fits, test_vols,iv_xfloor[txix] if not iv_xfloor is None else None, prev_vols if j>0 else None) ) 
                    else:
                        min_dx = vdetails.get("min_dx", 0.)
                        buf    = vdetails.get("buf", 0.)
                        ax.set_xlim( lo_k, hi_k )    
                        ax.set_ylim(  m_o_m( jr.iv_bids[det_ixs], jr.iv_asks[det_ixs], jr.iv_fits[det_ixs],\
                                            test_vols[det_ixs],iv_xfloor[det_xixs] if not iv_xfloor is None else None,
                                            prev_vols[det_ixs] if j>0 else None, min_dx=min_dx, buf=buf ) )
                        
                plot_vol( axV )
                plot_vol( axVD, vdetails )
    
                # plot density
                # ------------
        
                if not axD is None:
                    epsd = -1E-6
                    axD.plot( test_xstrikes, test_xstrikes*0., ":", color="black")
                    axD.plot( test_xstrikes[market_xp>=epsd], market_xp[market_xp>=epsd], "*:", color="orange", label="market")
                    axD.plot( test_xstrikes[market_xp<epsd], np.zeros_like(market_xp[market_xp<epsd]), "*", color="red", label="market (negative)") if sum(market_xp<0.) > 0 else None
                    axD.plot( test_xstrikes, test_xq, ".-", color="blue", label="interpolated")
                    
                    max_test_q = np.max( test_xq )
                    test_xix = ( test_xstrikes >= jr.strikes[0]-fdx ) & ( test_xstrikes <= jr.strikes[-1]+fdx )
                    axD.set_xlim( jr.strikes[0]-fdx, jr.strikes[-1]+fdx ) if not density_x else None
                    axD.set_ylim( m_o_m( np.minimum( max_test_q*2., market_xp[market_xp>=epsd] ), test_xq[test_xix] ) ) if not density_x else None
    
                    axD.set_xlabel("Extended strikes") if not axCA is None else None
                    axD.legend(loc="upper right") 
                    axD.set_xlabel("Strikes")

                if out_path is not None:
                    eta = None if self.is_linear else self.config.vol_fac**2
                    name_parts = [
                        "smoothvol" if not self.is_linear else "linearvol",
                    ]
                    if dt:
                        name_parts.append(dt)
                    name_parts.append(f"dte-{jr.label}")
                    if eta is not None:
                        name_parts.append(f"eta-{eta:.2f}")
                    fname = fmt_filename("_".join(name_parts)) + ".pdf"
                    fig.savefig(out_path / fname, silent_close=False, bbox_inches='tight')
    
    
    
            
    
    

