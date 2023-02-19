# -*- coding: utf-8 -*-
"""
Lite Black & Scholes FD engine
--
@author: hansbuehler
February 13 2023
"""

from .base import Logger, Config, np_unique_tol
from collections.abc import Mapping, Sequence # NOQA
import numpy as np
import math as math
import scipy.linalg as linalg

_log = Logger(__file__)

class Strip(object):
    
    def __init__(self, X : np.ndarray, F : np.ndarray, t : float ):
        self.X = X
        self.F = F
        self.t = t
        
    @property
    def fd_delta(self):
        """ First order derivative approximation of 'F'. Returns tuple x, delta """
        if len(self.X) < 3:
            return None, None
        delta = ( self.F[2:] - self.F[:-2] ) / ( self.X[2:] - self.X[:-2] )
        return self.X[1:-1], delta

    @property
    def fd_gamma(self):
        """ Second order derivative approximation of 'F'. Returns tuple x, gamma """
        if len(self.X) < 3:
            return None, None
        ku     = self.X[2:]
        km     = self.X[1:-1]
        kd     = self.X[:-2]
        dku    = ku - km
        dkd    = km - kd
        w_u    = 2. / ( dku * ( ku - kd ) )
        w_d    = 2. / ( dkd * ( ku - kd ) )
        fu     = self.F[2:]
        fm     = self.F[1:-1]
        fd     = self.F[:-2]
        gamma  = fu * w_u + fd * w_d - fm * ( w_u + w_d )
        return self.X[1:-1], gamma
    
    def bump_delta( self, spots, dx ):
        """
        Returns delta computed using a 'dx' bump 
        A good nump size is vol*sqrt(dt)
        """
        xup = spots+dx
        xdn = spots-dx
        fup = np.interp( xup, self.X, self.F )
        fdn = np.interp( xdn, self.X, self.F )
        delta = ( fup - fdn ) / ( 2.*dx )
        return delta

    def bump_delta_gamma( self, spots, dx ):
        """
        Returns delta, gamma computed using a 'dx' bump 
        A good nump size is vol*sqrt(dt)
        """
        xup = spots+dx
        xdn = spots-dx
        f   = np.interp( spots, self.X, self.F )
        fup = np.interp( xup, self.X, self.F )
        fdn = np.interp( xdn, self.X, self.F )
        delta = ( fup - fdn ) / ( 2.*dx )
        gamma = ( fup - 2*f + fdn ) / (dx**2)
        return delta, gamma

def bs_fd( *, spots : list, times : np.array, payoff, vol : float = 0.2, cn_factor = "implicit" ) -> list:
    """
    Finite difference solver for American and Barrier options wuth simple Black & Scholes
    The solver is a classic crank-nicolson solver over a non-homogeneous grid.
    
    Parameters
    ----------
        spots[nSteps+1]  : list of spots per time step. These spots must be widening. 
                           Alternatively, a numpy array spots[nSpots,nSteps+1] can be provided.
        times[nSteps+1]  : times
        payoff(X, F, t)  : compute new value F for spots X at time t. 'F' represents the current value and is None at maturity.
        vol              : BS vol
        cn_factor        : crank-nicolson factor. 0 for fully implicit, 1 for fully explicit
                           you can also use 'implicit' or 'explicit'

    Returns
    -------
        path
            path: a list of Strip's with:
                        t : time
                        X : spot
                        F : function
                        delta : F'
                        gamma : F''
            The first entry is the in  itial value at times[0] 

                                  
    Diffusion:
      dX_t  = X_t \sigma^2 dW_t
      
      E[dX^2] = X^2_t { E[ e^{2 \sigma \sqrt{dt} Y - \sigma^2 dt} - 1 ]  }
              = X^2_t { e^{ - \sigma^2 dt } e^{ 2 \sigma^2 dt } - 1 }
              = X^2_t { e^{ \sigma^2 dt } - 1 }
              \appox
                X^2_t \sigma^2 dt
          
    Backward PDE
      0     = df + 1/2 E[dX^2] f'' 
      df    = - 1/2 E[dX^2] f'' 
    
    explicit FD
      f_{t-dt} = f_t + 1/2 E[dX^2] f_t''
    implicit FD
      f_{t-dt} - 1/2 f_{t-dt}'' = f_t   
    
    Discretization 
      k_u     = (ku+km)/2
      k_d     = (km+kd)/2
      f'(k_u) = { f(ku) - f(km) } / { ku-km }
      f'(k_d) = { f(km) - f(kd) } / { km-kd }
      f''(km) = { f'(k_u) - f'(k_d) } / { k_u - k_d } 
     =>
      f''( km ) =   f(ku) wu    wu = 2 / {ku-km}{ku-kd}
                +   f(kd) wd    wd = 2 / {km-kd}{ku-kd}
                -   f(km) (wu + wd)

    """
    if isinstance( spots, np.ndarray ):
        _log.verify( len(spots.shape) == 2, "'spots': if a numpy array is provied, it must have dimension 2. Found %ld", len(spots.shape) )
        n = spots.shape[1]
        spots = [ np_unique_tol( spots[:,t], tol=1E-8, is_sorted=False ) for t in range(n) ]
    
    # basics
    nSteps   = len(spots)-1
    times    = np.asarray( times )
    _log.verify( times.shape == (nSteps+1,), "'times': must have shape %s, found shape %s", (nSteps+1,), times.shape)
    if isinstance(cn_factor, str):
        if cn_factor == "implicit":
            cn_factor = 0.
        elif cn_factor == "explicit":
            cn_factor = 1.
        else:
            _log.throw("Unknown 'cn_factor' '%s'. Must be 'implicit', 'explicit', or a Crank-Nicolson factor.", cn_factor)
    else:
        cn_factor = float(cn_factor)
        _log.verify( cn_factor >= 0. and cn_factor <= 1., "'cn_factor' must be from [0,1]. Found %g", cn_factor )
    
    # compute terminal value
    X          = np.asarray( spots[-1] )
    F          = payoff( X=spots[-1], F=None, t=times[-1] )
    MI         = np.zeros((3, len(X)))
    output     = [ Strip(F=F, X=X, t=times[-1] ) ]

    for t in range(nSteps-1,-1,-1):
        _log.verify( len(X) >= 3, "spots[%ld] must be at least of length 3. Found %ld", t+1, len(X) )
        _log.verify( np.min(X[1:]-X[:-1]) > 1E-10, "spots[%ld] must be increasing", t+1 )
        dt     = times[t+1] - times[t]
            
        # compute transition operators
        ku     = X[2:]
        km     = X[1:-1]
        kd     = X[:-2]
        dku    = ku - km
        dkd    = km - kd
        w_u    = 2. / ( dku * ( ku - kd ) )
        w_d    = 2. / ( dkd * ( ku - kd ) )
        varXdt = math.exp( vol * vol * dt ) - 1.   # correct second order for large implicit steps
        xi     = 0.5 * (km**2) * varXdt
        xi_u   = w_u * xi
        xi_d   = w_d * xi

        # explicit
        #  f_{t-dt} = ME * f_t
        #           
        #             1
        #             xi_d[0] 1-. xi_u[0]
        #     ME =           :
        #                     xi_d[-1] 1-. xi_u[-1]
        #                                 1
        #
        # we compoute this explicitly without constructing 'ME'.
        
        if cn_factor > 0.:
            xi1_m    = 1. - ( xi_u + xi_d ) * cn_factor
            xi__u    = cn_factor * xi_u
            xi__d    = cn_factor * xi_d
            # the boundary condition F_[b] = F[b] could be rather wrong ... 
            # formally it assumes that the outer derivative equals the inner
            # derivative and therefore that the value of F does not change.
            # That is correct if F is far out enough to be linear.
            F[1:-1]  = xi__u * F[2:] + xi__d * F[:-2] + xi1_m * F[1:-1]

        # implicit
        #  MI f_{t-dt} = f_t
        #
        #             1
        #             -xi_d[0] 1+. -xi_u[0]
        #     MI =           :
        #                     -xi_d[-1] 1+. -xi_u[-1]
        #                                    1
        if cn_factor < 1.:
            MI         = np.zeros((3,len(X))) if MI.shape[1] != len(X) else MI
            MI[1,1:-1] = 1. + (xi_u + xi_d ) * (1. - cn_factor)
            MI[1,0]    = 1.
            MI[1,-1]   = 1.
            MI[0,2:]   = - xi_u * (1. - cn_factor)
            MI[2,:-2]  = - xi_d * (1. - cn_factor)
            """
                # solve tridiag with matrix. For debugging.
                MI = np.zeros((nx,nx))
                for i in range(0,nx):
                    if i > 0 and i < nx-1:
                        MI[i,i]   = 1. + (xi_u[i-1] + xi_d[i-1] ) * (1. - cn_factor)
                        MI[i,i-1] = - xi_d[i-1] * (1. - cn_factor)
                        MI[i,i+1] = - xi_u[i-1] * (1. - cn_factor)
                    else:
                        MI[i,i] = 1.
                F = np.dot( np.linalg.inv( MI ), F )
                assert F.shape == (nx,), "Error: %s != %s" % ( F.shape, (nx,) )
            """
            F          = linalg.solve_banded( (1,1), MI, F )
            
        # interpolate to next spots
        # see https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2642630
        X_   = np.asarray( spots[t] )
                        
        if len(X) != len(X_) or np.max( np.abs(X-X_) ) > 1E-8:
            _log.verify_warn( X_[0] >= X[0]-1E-8, "spots[%ld][0] must not be less than spots[%ld][0]. Found %g and %g, respectively", t,t+1,X_[0],X[0] )
            _log.verify_warn( X_[-1] <= X[-1]+1E-8, "spots[%ld][-1] must not be greater than spots[%ld][-1]. Found %g and %g, respectively", t,t+1,X_[-1],X[-1] )
            F    = np.interp(X_, X, F )
        X    = X_

        # compute exercise and/or barrier 
        F      = payoff( X, F=F, t=times[t] )
        assert np.isfinite(F).all(), "'f' is not finite: %s" % F
        output.append( Strip(F=F, X=X, t=times[t]) )
    
    # return final functional value and 
    output.reverse()                        
    return output
