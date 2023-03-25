# -*- coding: utf-8 -*-
"""
Deep Hedging Objectives
-----------------------
Objectives, mostly monetary utilities see forthcoming book learning-to-trade.com
June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype, tfCast, fmt_list, DIM_DUMMY, perct_exp, fmt_list
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
    
    UTILITIES = ['mean', 'exp', 'exp2', 'vicky', 'cvar', 'quad']
    
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
        self.utility      = config("utility","exp2", self.UTILITIES, help="Type of monetary utility")
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
        return tf_utility(self.utility, self.lmbda, X, y=y )
        
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
    
    def compute_stateless_utility(self, payoff, sample_weights=None, **minimize_scalar_kwargs ) -> tf.Tensor:
        """
        Computes the utility of a payoff with classic optimization.
        This function only works if 'y' is stateless.
        Note that this function is not very accurate. Use trainer.train_utillity() for a more accurate version using tensorflow.
        
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
            _ = self.y(data={DIM_DUMMY:(payoff*0.)[:,np.newaxis]})
        except KeyError as k:
            _log.verify( self.nFeatures == 0, "Utility intercept 'y' relies on %ld features %s. Cannot compute simple initial utility. Use TensorFlow.", self.nFeatures,self.features )
        return oce_utility( self.utility, self.lmbda, X=payoff, sample_weights=sample_weights )

    
@tf.function  
def tf_utility( utility : str, lmbda : float, X : tf.Tensor, y : tf.Tensor = 0. ) -> dict:
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
        # Here p is small to reflect risk-aversion, e.g p=5% means we are computing the mean over the five worst percentiles.
        # Note that CVaR is often quoted with a survival percentile, e.g. q = 1-p e.g. 95%
        #
        # Conversion from percentile p (e.g. 5%) 
        #   1+lambda = 1/p 
        # =>
        #   lambda = 1/p - 1
        #
        # Conversion from lambda to percentile
        #   p = 1/(1+lambda)
        #
        # In other words, for p=50% use 1. (as in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3120710)
        #                 for p=5%  use 19.
        
        u = (1.+lmbda) * tf.math.minimum( 0., gains ) - y
        d = tf.where( gains < 0., -(1.+lmbda), 0. )

    elif utility == "quad":
        # quadratic penalty; flat extrapolation
        #
        # u(x)  = -0.5 lambda * ( x - x0 )^2 + 0.5 * lmbda * x0^2;   u(0)  = 0
        # u'(x) = - lambda (x-x0);                                   u'(0) = 1 = lambda x0 => x0 = 1/lmbda            
        
        x0 = 1./lmbda
        xx = tf.minimum( 0., gains-x0 )
        u  = - 0.5 * lmbda * (xx**2) + 0.5 * lmbda * (x0**2) - y
        d  = - lmbda * xx
                
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
        
    _log.verify( not u is None, "Unknown utility function '%s'. Use one of %s", utility, fmt_list( MonetaryUtility.UTILITIES )  )
    
    u = tf.debugging.check_numerics(u, "Numerical error computing u in %s. Turn on tf.enable_check_numerics to find the root cause.\nX: %s\ny : %s" % (__file__, str(X), str(y)) )
    d = tf.debugging.check_numerics(d, "Numerical error computing d in %s. Turn on tf.enable_check_numerics to find the root cause.\nX: %s\ny : %s" % (__file__, str(X), str(y)) )
    
    return pdct(
            u = u,
            d = d
        )
    
# -------------------------------------------------------------------------
# Mini OCE solver
# -------------------------------------------------------------------------
    
class _Objective( tf.keras.Model ):
    
    def __init__(self, utility, lmbda, init = 0. ):
        tf.keras.Model.__init__(self)
        self.utility = utility
        self.lmbda   = lmbda
        self.y       = tf.Variable(init, dtype=dh_dtype)
        
    def call(self, data, training=False):
        assert len(data.shape) == 1, "'data' must be a vector, found shape %s" % (data.shape.as_list())
        return -tf_utility( self.utility, self.lmbda, X=data, y=self.y ).u


def _default_loss( y_true,y_pred ):     
    """ Default loss: ignore y_true """
    return y_pred

def oce_utility( utility : str, lmbda : float, X : np.ndarray, sample_weights : np.ndarray = None, method : str = None, epochs : int = 100, batch_size : int = 'all', **minimize_scalar_kwargs ) -> float:
    """
    Stand-alone OCE utility calculation, using analytical solutions where possible or a numerical 
    
    Parameters
    ----------
        utility:
            Name of the utility function, see MonetaryUtility.UTILITIES
        lmbda:
            Risk aversion
        X:
            Variable to compute utility for
        sample_weights:
            Sammple weights or None for 1/n
        method:
            None for best,
            'minscalar' for numerical minimization
            'tf' for tensorflow
        epochs, batch_size:
            For tensorflow mode.
            Use batch_size='all' for full batch size, None for 32 or a specific number
        minimize_scalar_kwargs:
            Arguments for 'minscalar'
            
    Returns
    -------
        Result
    """
    
    lmbda          = float(lmbda)
    X              = np.asarray(X)
    sample_weights = np.asarray(sample_weights) if not sample_weights is None else None

    _log.verify( len(X.shape) == 1, "'X' must be a vector, found shape %s", X.shape)

    if not sample_weights is None:
        _log.verify( sample_weights.shape[0] == len(X), "'sample_weights' first dimension must be %ld, not %ld", len(X), sample_weights.shape[0] )
        if len(sample_weights.shape) == 2:
            _log.verify( sample_weights.shape[1] == 1, "'sample_weights' second dimension must be 1, not %ld", sample_weights.shape[1] )
        else:
            _log.verify( len(sample_weights.shape) == 1, "'sample_weights' must be vector of length %ld. Found shape %ld", len(X), sample_weights.shape )
            sample_weights = sample_weights[:,np.newaxis]
        sample_weights   /= np.sum( sample_weights )
    
    # analytical?
    # -----------

    if method is None:
        P = sample_weights[:,0] if not sample_weights is None else None
        
        if utility in ["mean", "expectation"] or lmbda == 0.:
            return np.sum(P * X) if not P is None else np.mean(X)
        
        if utility in ["exp", "entropy"]:
            expX = np.exp( - lmbda * X )
            eexp = np.sum( P * expX ) if not P is None else np.mean( expX )
            return - np.log( eexp ) / lmbda

        if utility == "cvar":
            p         = 1./(1. + lmbda)
            assert p>0. and p<=1., "Invalid percentile %g" % p
            if abs(1.-p)<1E-8:
                return np.sum(sample_weigths * X) if not sample_weigths is None else np.mean(X)

            if sample_weights is None:
                pcnt     = np.percentile( X, p*100. )
                ixs      = X <= pcnt
                return np.mean( X[ixs] )

            ixs       = np.argsort(X)
            X         = X[ixs]
            P         = P[ixs]
            cumP      = np.cumsum(P)
            assert abs(cumP[-1]-1.)<1E-4, "Internal error: cumsum(P)[-1]-1 = %g" % (cumP[-1]-1.)
            cumP[-1]  = 1.
            if p <= cumP[0]:
                return X[0]
            ix        = np.searchsorted( cumP, p )
            ix        = max( 0., min( len(X)-1, ix ))
            return np.sum( (P * X)[:ix+1] ) / np.sum( P[:ix+1] )
        
    # minimize_scalar
    # ---------------
    
    if method is None or method == 'minscalar':
        X = tfCast(X, dtype=dh_dtype)
        def objective(y):
            y = np.asarray(y)
            y = tfCast(y, dtype=dh_dtype)
            r = tf_utility(utility, lmbda, X, y=y )
            u = np.asarray( r.u )
            u = np.sum( sample_weights[:,0] * u ) if not sample_weights is None else np.mean(u)
            return -u

        _  = objective(0.)   # triggers errors if any
        r  = minimize_scalar( objective, tol=1E-6, **minimize_scalar_kwargs )
        if not r.success: _log.error( "Failed to find optimal intercept 'y' for utility %s with risk aversion %g: %s", utility, lmbda, r.message )
        return -objective(r.x)

    # tensorflow
    # ----------

    _log.verify( method == 'tf', "'method' must be None, 'minscalar', or 'tf'. Found %s", method )
    
    batch_size     = len(X) if batch_size == 'all' else batch_size
    epochs         = int(epochs)
    X              = tfCast(X, dtype=dh_dtype)
    sample_weights = tfCast(sample_weights, dtype=dh_dtype) if not sample_weights is None else None

    model = _Objective( utility, lmbda, np.mean(X) )
    model.compile( optimizer = "adam", loss=_default_loss )
    model.fit(  x              = X,
                y              = X*0.,
                batch_size     = batch_size,
                sample_weight  = sample_weights * float(len(X)) if not sample_weights is None else None,  # sample_weights are poorly handled in TF
                epochs         = epochs,
                verbose = 0)

    return -np.mean(model(X))
