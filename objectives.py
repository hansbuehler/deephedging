# -*- coding: utf-8 -*-
"""
Deep Hedging Objectives
-----------------------
Objectives, mostly monetary utilities see forthcoming book learning-to-trade.com
June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype, tfCast, fmt_list
from .layers import DenseLayer, VariableLayer
from cdxbasics import PrettyDict as pdct
from collections.abc import Mapping
from scipy.optimize import minimize_scalar
import numpy as np

_log = Logger(__file__)
 
class MonetaryUtility(tf.keras.Model):
    """
    Monetary utility function as standard objective for deep hedging.
    The objective for a claim X is defined as
    
        sup_y E[ u(X+y)-y ]
 
    The variable 'y' needs to be learned.
    The standard implementation is to learn the variable for the intial payoff and the overall hedging gains in the same
    training loop (see loss definition for the gym).

    By default 'y' is a plain real variable. That assumes that the initial state of the world is constant, and does not
    differ by path. An example to the contrary is if we wanted to learn a hedging strategy accross products (e.g. different strikes for calls).
    In this case, 'y' would have to be a network which depends on such `per_path` features.
    The list of available per path features for a given world in a given gym can be obtained using gym.available_features_per_path()

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
        tf.keras.Model.__init__(self, name=name, dtype=dtype )
        self.utility      = config("utility","exp2", ['mean', 'exp', 'exp2', 'vicky', 'cvar', 'quad'], help="Type of monetary utility")
        self.lmbda        = config("lmbda", 1., float, help="Risk aversion")
        self.display_name = self.utility + "@%g" % self.lmbda
        _log.verify( self.lmbda > 0., "'lmnda' must be positive. Use utility 'mean' for zero lambda")
        
        if self.utility in ["mean"]:
            _log.warning("Using utility mean - OCE 'y' is now fixed.")
            self.y       = VariableLayer( 0., trainable=False, name=name+"_OCE_y_fixed" if not name is None else "OCE_y_fixed", dtype=dtype )
            config.y.mark_done()  # avoid error message from config.done()
        else:       
            features     = config.y("features", [], list, "Path-wise features used to define 'y'. If left empty, then 'y' becomes a simple variable.")
            self.y       = DenseLayer( features=features, nOutput=1, initial_value=0., config=config.y.network, name= name+"_OCE_y" if not name is None else "OCE_y", dtype=dtype )
        config.done() # all config read
        
    def call( self, data : dict, training : bool = False ) -> tf.Tensor:
        """
        Compute the monetary utility for a Deep Hedging problem.
        
        Parameters
        ----------
            data : dict
                A dictrionary of tensors with all features available
                at time zero. All tensors mus thave dimension 2.
                Expects
                    features_time_0 : all features available at time zero (see comments in the class description)
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
        return self.compute( X = X, features_time_0 = features, training=training )['u']
        
    def compute( self, X : tf.Tensor, features_time_0 : dict = None, training : bool = False ) -> dict:
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
        training : bool, optional
            Whether we are in training model

        Returns
        -------
            dict: 
                Contains 'u' and 'd', the utility and its derivative
        """ 
        _log.verify( isinstance(features_time_0, Mapping), "'features_time_0' must be a dictionary type. Found type %s", type(features_time_0))
        features_time_0 = features_time_0 if not features_time_0 is None else {}
        y      = self.y( features_time_0, training=training )
        assert len(y.shape) == 2 and y.shape[1] == 1, "Internal error: expected variable to return a vector of shape [None,1]. Found %s" % y.shape.as_list()
        y      = y[:,0]
        #y     = tf.debugging.check_numerics(y, "Numerical error computing OCE_y in %s" % __file__ )
        return utility(self.utility, self.lmbda, X, y=y )
        
    @property
    def features(self):
        """ Features used by the utility """
        return self.y.features
    @property
    def available_features(self):
        """ Features available to the utility. """
        return self.y.available_features
    @property
    def nFeatures(self):
        """ Number of features used """
        return self.y.nFeatures
    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] )
    @property
    def description(self):
        """ Returns a text description of 'self' """
        text =        "Monetary utility %s is using %ld weight%s" % (self.display_name, self.num_trainable_weights, "s" if self.num_trainable_weights != 1 else "")
        text +=     "\n Features available: %s" % fmt_list( self.y.available_features )
        text +=     "\n Features used:      %s" % fmt_list( self.y.features )
        return text
    
    # -----------------------------------
    # Analytical
    # -----------------------------------
    
    def compute_stateless_utility(self, payoff, sample_weights=None, **minimize_scalar_kwargs ) -> tf.Tensor:
        """
        Computes the utility of a payoff with classic optimization.
        This function only works if 'y' is stateless.
        
        Parameters
        ----------
            payoff : the terminal value to compute a utility for
            sample_weights: from the world. If None, set to 1/N
            
        Returns
        -------
            The utility value as a float.
        """
        
        try:
            # trigger build(). This will fail if 'y' expects some features 
            _ = self.y(data={})
        except KeyError as k:
            _log.verify( self.nFeatures == 0, "Utility intercept 'y' relies on %ld features %s. Cannot compute simple initial utility. Use TensorFlow.", self.nFeatures,self.features )

        payoff           = np.array( payoff )
        np_dtype         = payoff.dtype
        _log.verify( len(payoff.shape) == 1, "'payoff' must have shape of dimension 1. Found shape %s", payoff.shape )        
        nSamples         = int( payoff.shape[0] )
        sample_weights   = np.array( sample_weights, dtype=np_dtype ) if not sample_weights is None else np.full( (nSamples,), 1./float(nSamples), dtype=np_dtype )
        sample_weights   = sample_weights[:,0] if not sample_weights is None and len(sample_weights.shape) == 2 and sample_weights.shape[1] == 1 else sample_weights
        
        _log.verify( payoff.shape == sample_weights.shape, "'payoff' must have same shape as 'sample_weights'. Found %s and %s, respectively", payoff.shape, sample_weights.shape )
        
        def objective(y):
            y = np.array(y, dtype=np_dtype)
            r = utility(self.utility, self.lmbda, tf.convert_to_tensor(payoff), y=tf.convert_to_tensor(y) )
            u = np.array( r.u, dtype=np_dtype )
            u = np.sum( sample_weights * u )
            return -u

        r0 = -objective(0.)  # will throw any errors
        r  = minimize_scalar( objective, **minimize_scalar_kwargs )
        _log.verify( r.success, "Failed to find optimal intercept 'y' for utility %s with risk aversion %g: %s", self.utility, self.lmbda, r.message )
        return -r.x
        
        
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
        # Zero lambda => mean
        utility = "mean"
        lmbda   = 1.

    if utility in ["mean", "expectation"]:
        # Expectation
        #
        u = gains
        d = tf.ones_like(gains)
        
    elif utility == "cvar":
        # CVaR risk measure
        #   u(x) = (1+lambda) min(0, x)
        # The resulting OCE measure U computes the expected value under the condition that X is below the p's percentile.
        #   U(X) = E[ X | X <= P^{-1}[ X<=* ](p)
        #
        # Conversion from percentile p (e.g. 95%):
        #   1+lambda = 1/(1-p) 
        # and
        #   lambda = p / (1-p)
        #
        # In other words, for p=50% use 1. (as in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3120710)
        #                 for p=95% use 19.
        
        u = (1.+lmbda) * tf.math.minimum( 0., gains ) - y
        d = tf.where( gains < 0., -(1.+lmbda), 0. )

    elif utility == "quad":
        # quadratic penalty; flat extrapolation
        #
        # u(x)  = -0.5 lambda * ( gains - x0 )^2 + 0.5 * x0^2;   u(0)  = 0
        # u'(x) = - lambda (gains-x0);                           u'(1) = lambda x0 => x0 = 1/lmbda            
        
        x0 = 1./lmbda            
        u  = tf.where( gains < x0, - 0.5 * lmbda * ( ( gains - x0 ) ** 2 ), 0. ) + 0.5 * (x0**2) - y
        d  = tf.where( gains < x0, - lmbda * (gains - x0), 0. ) 
                
    elif utility in ["exp", "entropy"]:
        # Entropy
        #   u(x) = { 1 - exp(- lambda x ) } / lambda 
        #
        # The OCE measure for this utility has the closed form
        #   U(X) = - 1/lambda log E[ exp(-\lambda X) ]
        #
        # However, this tends to be numerically challenging.
        # we introcue a robust version less likely to explode
        inf = tf.stop_gradient( tf.reduce_min( X ) )
        u = (1. - tf.math.exp( - lmbda * (gains-inf)) ) / lmbda - y + inf
        d = tf.math.exp(- lmbda * gains )
        
    elif utility == "exp2":
        # Exponential for the positive axis, quadratic for the negative axis.
        # A non-exploding version of the entropy
        #
        # u1(x)  = { 1-exp(-lambda x) } / lambda; u1(0)  = 0 
        # u1'(x) = exp(-lambda x);                u1'(0) = 1       
        # u2(x)  = x - 0.5 lambda x^2;            u2(0)  = 0
        # u2'(x) = 1 - lambda x;                  u2'(0) = 1
        g1  = tf.maximum(gains,0.)
        g2  = tf.minimum(gains,0.)
        eg1 = tf.math.exp( - lmbda * g1)
        u1  = (1. - eg1 ) / lmbda - y            
        u2  = g2 - 0.5 * lmbda * g2 * g2 - y
        d1  = eg1
        d2  = 1. - lmbda * g2
        u   = tf.where( gains > 0., u1, u2 )
        d   = tf.where( gains > 0., d1, d2 )
        
    elif utility == "vicky":
        # Vicky Handerson & Mark Rodgers
        # https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/henderson/publications/indifference_survey.pdf
        #
        # u(x)  = { 1 + lambda * x - sqrt{ 1+lambda^2*x^2 } } / lmbda
        # u'(x) = 1 - lambda x / sqrt{1+lambda^2*x^2}
        u = (1. + lmbda * gains - tf.math.sqrt( 1. + (lmbda * gains) ** 2 )) / lmbda  - y
        d = 1 - lmbda * gains / tf.math.sqrt( 1. + (lmbda * gains) ** 2)
        
    _log.verify( not u is None, "Unknown utility function '%s'", utility )      
    
    u = tf.debugging.check_numerics(u, "Numerical error computing u in %s. Turn on tf.enable_check_numerics to find the root cause.\nX: %s\ny : %s" % (__file__, str(X), str(y)) )
    d = tf.debugging.check_numerics(d, "Numerical error computing d in %s. Turn on tf.enable_check_numerics to find the root cause.\nX: %s\ny : %s" % (__file__, str(X), str(y)) )
    
    return pdct(
            u = u,
            d = d
        )
    

    
  

        
        