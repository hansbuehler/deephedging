# -*- coding: utf-8 -*-
"""
Deep Hedging Objectives
-----------------------
Objectives, mostly monetary utilities see forthcoming book learning-to-trade.com
June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype
from .agents import AgentFactory
from cdxbasics import PrettyDict as pdct
from collections.abc import Mapping
_log = Logger(__file__)
 
class MonetaryUtility(tf.keras.layers.Layer):
    """
    Monetary utility function as standard objective for deep hedging.
    The objective for a claim X is defined as
    
        sup_y E[ u(X+y)-y ]
        
    Attributes
    ----------
        y_model : bool
            Whether the intercept 'y' is a model which requires features
        features : list
            List of features required; possibly [] if 'y' is a variable or 0.
    
    Members
    -------
        __call__()
            Tensor flow call to evaluate the utility for a given environment
        compute()
            Computes utility and its derivative after training.
            
    Hans Buehler, June 2022
    """
    
    Default_features = [ 'price', 'delta', 'time_left' ]

    def __init__(self, config : Config, name : str = None, dtype : tf.DType = dh_dtype ):
        """
        Parameters
        ----------
            config : Config
                configuration, most notably            
                utility  - which utility to use e.g. mean, exp, vicky, quad
                lmbda    - risk aversion
                features - features to use for time 0 y.
                           Leave empty for a determinstic y amount
                
            name : str, optional
                Name of the tenosrflow model
            dtype : tf.DType, optional
                dtype
        """
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )
        self.utility    = config("utility","entropy", str, help="Type of monetary utility: mean, exp, exp2, vicky, cvar, quad")
        self.lmbda      = config("lmbda", 1., float, help="Risk aversion")
        self.y_model    = False
        _log.verify( self.lmbda > 0., "'lmnda' must be positive. Use utility 'mean' for zero lambda")
        
        if self.utility in ["mean"]:
            self.y       = tf.Variable( 0., trainable=False )
            config.y.mark_done()  # avoid error message from config.done()
        elif config.y.is_empty:
            self.y       = tf.Variable( 0., trainable=True, name="OCE_y" )
        else:       
            self.y       = AgentFactory( 1, config.y, dtype=dtype )
            self.y_model = True
        config.done() # all config read
        
    @property
    def features(self) -> list:
        """ Returns list of features required by this utility """
        return self.y.features if self.y_model else []
        
    def call( self, data : dict, training : bool = False ) -> tf.Tensor:
        """
        Compute the monetary utility for a Deep Hedging problem.
        
        Parameters
        ----------
            data : dict
                A dictrionary of tensors with all features available
                at time zero. All tensors mus thave dimension 2.
                Expects
                		features_time_0 : all features required at this time, c.f. what was provided to init()
                		payoff          : [nSamples,] terminal payoff
                		pnl             : [nSamples,] trading pnl
                		cost            : [nSamples,] cost.
               OCE utilities operate on X := payoff + gains - cost
            training : bool, optional
                See tensor flow documentation
                
        Returns
        -------
            The utility value, per path.
        """
        features = data['features_time_0']
        payoff   = data['payoff']
        pnl      = data['pnl']
        cost     = data['cost']
        X        = payoff + pnl - cost
        return self.compute( X = X, features_time_0 = features )['u']
        
    def compute( self, X : tf.Tensor, features_time_0 : dict = None ) -> dict:
        """
        Computes
            u(X+y) - y
        and its derivative in X for random variable X and y=self.y
    				
        Parameters
        ----------
        X: tf.Tensor
            Random variable, typically total gains on the path
        features_time_0 : dict
            features required for 'y' if so specified.
            Check self.features
    			
        Returns
        -------
            dict: 
                Contains 'u' and 'd', the utility and its derivative
        """ 
        _log.verify( isinstance(features_time_0, Mapping), "'features_time_0' must be a dictionary type. Found type %s", type(features_time_0))
        features_time_0 = features_time_0 if not features_time_0 is None else {}
        y               = self.y( features_time_0 ) if self.y_model else self.y
        return utility(self.utility, self.lmbda, X, y=y )
        
@tf.function  
def utility( utility : str, lmbda : float, X : tf.Tensor, y : tf.Tensor = 0. ) -> dict:
    """
    Computes
        u(X+y) - y
    and its derivative in X for random variable X and OCE variable y 

    Parameters
    ----------
    utility: str
        Which utility function 'u' to use
    lmbda : flost
        risk aversion
    X: tf.Tensor
        Random variable, typically total gains on the path
    y: tf.Tensor, None, or 0
        OCE intercept y.
			
    Returns
    -------
        dict:
            with menbers 'u' and 'd'
    """
    utility  = str(utility)
    lmbda    = float(lmbda)
    y        = y if not y is None else 0.
    gains    = X + y	
    
    _log.verify( lmbda >= 0., "Risk aversion 'lmbda' cannot be negative. Found %g", lmbda )
    if lmbda < 1E-12: 
        utility = "mean"
        lmbda   = 1.

    if utility in ["mean", "expectation"]:
        u = gains
        d = tf.ones_like(gains)
        
    elif utility == "cvar":
        # CVar risk measure.
        # 1+lambda = 1/(1-p) where p is the required percentile, e.g. 95%
        # For a given percentile
        #   lambda = p / (1-p)
        # For a give lmbda
        #   p = L / (1+L)
        u = (1.+lmbda) * tf.math.minimum( gains ) - y
        d = tf.where( gains < 0., -(1.+lmbda), 0. )

    elif utility == "quad":
        # quadratic CVaR: quadratic penalty; flat extrapolation
        # u(x)  = -0.5 lambda * ( gains - x0 )^2 + 0.5 * x0^2;   u(0)  = 0
        # u'(x) = - lambda (gains-x0);                           u'(1) = lambda x0 => x0 = 1/lmbda            
        x0 = 1./lmbda            
        u  = tf.where( gains < x0, - 0.5 * lmbda * ( ( gains - x0 ) ** 2 ), 0. ) + 0.5 * (x0**2) - y
        d  = tf.where( gains < x0, - lmbda * (gains - x0), 0. ) 
                
    elif utility in ["exp", "entropy"]:
        # entropy.
        # { 1 - exp(- lambda x ) } / lambda 
        # we introcue a robust version less likely to explode
        inf = tf.stop_gradient( tf.reduce_min( gains ) )
        u = (1. - tf.math.exp( - lmbda * (gains-inf)) ) / lmbda - y + inf
        d = tf.math.exp(- lmbda * gains )
        
    elif utility == "exp2":
        # Exponential for the positive axis, quadratic for the 
        # negative axis
        # u1(x)  = { 1-exp(-lambda x) } / lambda; u1(0)  = 0 
        # u1'(x) = exp(-lambda x);                u1'(0) = 1       
        # u2(x)  = x - 0.5 lambda x^2;            u2(0)  = 0
        # u2'(x) = 1 - lambda x;                  u2'(0) = 1
        u1 = (1. - tf.math.exp( - lmbda * gains) ) / lmbda - y            
        u2 = gains - 0.5 * lmbda * gains * gains - y
        d1 = tf.math.exp(- lmbda * gains)
        d2 = 1. - lmbda * gains
        u  = tf.where( gains > 0., u1, u2 )
        d  = tf.where( gains > 0., d1, d2 )
        
    elif utility == "vicky":
        # Vicky Handerson & Mark Rodgers
        # u(x)  = { 1 + lambda * x - sqrt{ 1+lambda^2*x^2 } } / lmbda
        # u'(x) = 1 - lambda x / sqrt{1+lambda^2*x^2}
        # https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/henderson/publications/indifference_survey.pdf
        u = (1. + lmbda * gains - tf.math.sqrt( 1. + (lmbda * gains) ** 2 )) / lmbda  - y
        d = 1 - lmbda * gains / tf.math.sqrt( 1. + (lmbda * gains) ** 2)
        
    _log.verify( not u is None, "Unknown utility function '%s'", utility )      
    
    return pdct(
            u = u,
            d = d
        )
    
