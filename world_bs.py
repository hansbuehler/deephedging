# -*- coding: utf-8 -*-
"""
Deep Hedging Example Worlds
---------------------------
Example world for deep hedging.

June 30, 2022
@author: hansbuehler
"""

from deephedging.base import Logger, Config, dh_dtype, tf, tfCast, pdct, tf_dict, assert_iter_not_is_nan, Int, Float, DIM_DUMMY
from cdxbasics.dynaplot import figure, colors_tableau
import numpy as np
import math as math
#from tqdm import tqdm
from scipy.stats import norm
_log = Logger(__file__)

class SimpleWorld_BS(object):
    """
    Black Scholes, no gimmicks.
    
    Members
    -------
        clone()
            Create a clone of this world with a different seed as validation set
            
    Attributes
    ----------
        data : dict
            Numpy data of the world

            market : dict
                Dictionary of market data with second dimension equal to step size (numpy)

            features : dict
                per_step : dict - Dictionary of features with second dimension equal to step size (numpy)
                per_path : dict - Dictionary of features valid per path
                                            
        tf_data : dict
            Returns a dictionary of TF tensors of 'data' for the use of gym.call() or train()

        tf_y : tf.Tensor
            y data for gym.call() or train(), usually a dummy vector.
            
        sample_weights : np.ndarray
            sample weights for manual calculations outside tensorflow
            Dimension (nSamples,)
            
        tf_sample_weights : tf.Tensor:
            sample weights for train()
            Dimension (nSamples,1) c.f. https://stackoverflow.com/questions/60399983/how-to-create-and-use-weighted-metrics-in-keras

        details : dict
            Dictionary of details, e.g. the hidden drift and realized vol
            of the asset (numpy)

        nSamples : int
            Number of samples
            
        nSteps : int
            Number of steps
            
        nInst : int
            Number of instruments.
            
        dt : floast
            Time step. TODO: remove in favour of timeline
            
        timelime : np.ndarray
            Generalized timeline. Includes last time point T, e.g. is of length nSteps+1
            
        config : Config
            Copy of the config file, for cloning
            
        unique_id : str
            Unique ID generate off the config file, for serialization
    """
    
    def __init__(self, config : Config, dtype=dh_dtype ):
        """
        Parameters
        ----------
        config : Config
            Long list. Use the report feature of 'config' for full feature set
                config  = Config()
                world   = SimpleWorld_BS(config)
                print( config.usage_report( with_values = False )
                      
             To use black & scholes mode use hard overwrite black_scholes = True
        """
        self.dtype     = dtype
        self.unique_id = config.unique_id() # for serialization
        self.config    = config.copy()      # for cloning

        # Read config
        # -----------
        
        # basic MC settings
        nSteps     = config("steps", 10,        Int>0, "Number of time steps")
        nSamples   = config("samples", 1000,    Int>2, "Number of samples")
        seed       = config("seed", 2312414312, int,  "Random seed")
        dt         = config("dt", 1./50.,       Float>1./10000., "Time per timestep.", help_default="One week (1/50)")
        np.random.seed(seed)

        # black scholes parameters
        drift      = config("drift", 0.,        float, "Annualized real-life drift of the asset. Note that the risk-neutral drift is always zero")
        vol        = config("vol",   0.2,       Float>=0., "Annualized volatility of the asset")

        # basic DH parameters
        cost_s     = config("cost", 0.0002,     Float>=0., "Trading cost, as fractions of spot")
        ubnd_a     = config("ubnd_a", 5.,       Float>=0., "Upper bound for the number of shares traded at each time step")
        lbnd_a     = config("lbnd_a", -5.,      Float<=0., "Lower bound for the number of shares traded at each time step")
        _log.verify( ubnd_a - lbnd_a > 0., "'ubnd_a' (%g) must be bigger than 'lbnd_as' (%g)", ubnd_a, lbnd_a )

        # payoff
        # ------
        # must either be a function of spots[samples,steps+1], None, or a fixed umber
        payoff_f  = config("payoff", "atmcall", help="Payoff function with parameter spots[samples,steps+1]. Can be a function which must return a vector [samples]. Can also be short 'atmcall' or short 'atmput', or a fixed numnber. The default is 'atmcall' which is a short call with strike 1: '- np.maximum( spots[:,-1] - 1, 0. )'. A short forward starting ATM call is given as '- np.maximum( spots[:,-1] - spots[:,0], 0. )'.")
        if payoff_f is None:
            # None means zero.
            payoff_f = np.zeros( (nSamples, ) )
        elif isinstance(payoff_f, (int,float)):
            # specify terminal payoff as a fixed number, e.g. 0
            payoff_f = np.full( (nSamples,), float(payoff_f) )
        elif isinstance(payoff_f, str):
            if payoff_f == "atmcall":
                payoff_f = lambda spots : - np.maximum( spots[:,-1] - 1., 0. )
            elif payoff_f == "atmput":
                payoff_f = lambda spots : - np.maximum( 1. - spots[:,-1], 0. )
            else:
                _log.throw("Unknown 'payoff' '%s'", payoff_f)

        # -----------------------------
        # unique_id
        # -----------------------------
        # Default handling for configs will ignore any function definitions, e.g. in this case 'payoff'.
        # we therefore manually generate a sufficient hash
        self.unique_id = uniqueHash( [ config.input_dict(), payoff_f, self.tf_dtype.name ],parse_functions = True )

        config.done()
        
        # path generator
        # --------------
        
        # note that the 'spot' vector contains nStep+1 elements
        # beause we do not need to hedge at the last day
        sqrtDt      = math.sqrt(dt)
        dW          = np.random.normal(size=(nSamples,nSteps)) * sqrtDt
        dlogS       = drift * dt + vol * dW - 0.5 * vol * vol * dt
        spot        = np.ones((nSamples,nSteps+1,))
        spot[:,1:]  = np.exp( np.cumsum(dlogS, axis=1) )
        hedges      = spot[:,-1][:,tf.newaxis] - spot[:,:-1] # DH_t :+ S(T) - S(t)
        hedges      = hedges[:,:,np.newaxis]
        cost        = cost_s * spot[:,:-1][:,:,np.newaxis]
        time_left   = np.linspace(float(nSteps), 1., nSteps, endpoint=True) * dt
        sqrt_time_left = np.sqrt( time_left )
        ubnd_a      = np.full(hedges.shape, ubnd_a)
        lbnd_a      = np.full(hedges.shape, lbnd_a)

        # payoff
        # ------        
        # The payoff function may return either payoff per sample, or dictionary with 'payoff' and 'features'
        # The features are expected to be of dimension (nSamples, nSteps, n).
        
        if not isinstance(payoff_f, np.ndarray):
            payoff    = payoff_f( spot[:,:nSteps+2] )
            py_feat   = None
            if isinstance( payoff, Mapping ):
                py_feat = np.asarray( payoff['features'] )
                payoff  = np.asarray( payoff['payoff'] )
                
                _log.verify( len(py_feat.shape) in [2,3], "payoff['features']: must have dimension 2 or 3, found %ld", len(py_feat.shape) )
                _log.verify( py_feat.shape[0] == nSamples and py_feat.shape[1] == nSteps, "payoff['features']: first two dimension must be (%ld,%ld). Found (%ld, %ld)", nSamples, nSteps, py_feat.shape[0], py_feat.shape[1] )
                py_feat    = py_feat[:,0] if len(py_feat) == 2 else py_feat
            else:
                payoff  = np.asarray( payoff )
            
            payoff    = payoff[:,0] if payoff.shape == (nSamples,1) else payoff
            _log.verify( payoff.shape == (nSamples,), "'payoff' function which receives a vector spots[nSamples,nSteps+1] must return a vector of size nSamples. Found shape %s", payoff.shape )
        else:
            _log.verify( payoff_f.shape == (nSamples,), "'payoff' if a vector is provided, its size must match the sample size. Expected shape %s, found %s", (nSamples,), payoff_f.shape )
            payoff     = payoff_f
            py_feat    = None
        
        # -----------------------------
        # store data
        # -----------------------------
        
        # market
        # note that market variables are *not* automatically features
        # as trhey often look ahead
        
        self.data = pdct()
        self.data.market = pdct(
                hedges    = hedges,
                cost      = cost,
                ubnd_a    = ubnd_a, 
                lbnd_a    = lbnd_a,
                payoff    = payoff
            )
                
        # features
        # observable variables for the agent
        self.data.features = pdct(
            per_step = pdct(
                # both spot and option, if present
                cost           = cost,            # trading cost
                spot           = spot[:,:-1],     # price of the hedge
                price          = spot[:,:-1],     # price of the hedge
                ubnd_a         = ubnd_a,          # bounds. Currently those are determinstic so don't use as features
                lbnd_a         = lbnd_a,
                time_left      = np.full( (nSamples, nSteps), time_left[np.newaxis,:] ),
                sqrt_time_left = np.full( (nSamples, nSteps), sqrt_time_left[np.newaxis,:] ),
                ),
            per_path = pdct(),
            )
        if not py_feat is None:
            self.data.features.per_step['payoff_features'] = py_feat
        
        # the following variables must always be present in any world
        # it allows to cast dimensionless variables to the number of samples
        self.data.features.per_path[DIM_DUMMY] = (payoff * 0.)[:,np.newaxis]  # (None,1)
            
        # check numerics
        assert_iter_not_is_nan( self.data, "data" )
 
        # tf_data
        # what gym() gets
        
        self.tf_data = tf_dict( 
            features = self.data.features,
            market   = self.data.market,
            )
    
        # details
        # variables for visualization, but not available for the agent
        # TODO: remov dependency on this by plotting
        self.details = pdct(
            spot_all = spot, # [nSamples,nSteps+1] spots including spot at T
            )
    
    
        # check numerics
        assert_iter_not_is_nan( self.details, "details" )
        
        # generating sample weights
        # the tf_sample_weights is passed to keras train and must be of size [nSamples,1]
        # https://stackoverflow.com/questions/60399983/how-to-create-and-use-weighted-metrics-in-keras
        self.sample_weights    = np.full((nSamples,1),1./float(nSamples))
        self.tf_sample_weights = tf.constant( self.sample_weights, dtype=self.dtype)  # must be of size (nSamples,1)
        self.sample_weights    = self.sample_weights.reshape((nSamples,))             # for numpy usage, set size (nSamples,)
        self.tf_y              = tf.zeros((nSamples,), dtype=self.dtype)
        self.nSteps            = nSteps
        self.nSamples          = nSamples
        self.nInst             = 1
        self.inst_names        = [ 'spot' ]
        self.dt                = dt
        self.timeline  = np.cumsum( np.linspace( 0., nSteps, nSteps+1, endpoint=True, dtype=np.float32 ) ) * dt
        
    def clone(self, config_overwrite = Config(), **kwargs ):
        """
        Create a copy of this world with the same config, except for the seed.
        Used to generate genuine validation sets.
        
        Parameters
        ----------
            config_overwrite : Config, optional
                Allows specifying additional overwrites of specific config values
            **kwargs
                Allows specifying additional overwrites of specific config values, e.g.
                    world.clone( seed=222, samples=10 )
                If seed is not specified, a random seed is generated.

        Returns
        -------
            New world
        """
        if not 'seed' in kwargs:
            kwargs['seed'] = int(np.random.randint(0,0x7FFFFFFF))
        config = self.config.copy()
        config.update( config_overwrite, **kwargs )        
        return SimpleWorld_BS( config )

    def plot(self, config = Config(), **kwargs ):
        """ Plot simple world """
        
        config.update(kwargs)
        col_size     = config.fig("col_size", 5, int, "Figure column size")
        row_size     = config.fig("row_size", 5, int, "Figure row size")
        plot_samples = config("plot_samples", 5, int, "Number of samples to plot")
        print_input  = config("print_input", True, bool, "Whether to print the config inputs for the world")
        
        xSamples     = np.linspace(0,self.nSamples,plot_samples,endpoint=False, dtype=int)
        timeline1    = np.cumsum( np.linspace( 0., self.nSteps, self.nSteps+1, endpoint=True, dtype=np.float32 ) ) * self.dt
        timeline     = timeline1[:-1]
        
        
        print(self.config.usage_report())
        
        fig = figure(tight=True, col_size=col_size, row_size=row_size, col_nums=3 )
        fig.suptitle(self.__class__.__name__, fontsize=16)
        
        # spot
        ax  = fig.add_plot()
        ax.set_title("Spot")
        ax.set_xlabel("Time")
        for i, color in zip( xSamples, colors_tableau() ):
            ax.plot( timeline1, self.details.spot_all[i,:], "-", color=color )
        ax.plot( timeline1, np.mean( self.details.spot_all, axis=0), "_", color="black", label="mean" )
        ax.legend()
        
        fig.render()
        fig.close()

        if print_input:
            print("Config settings:\n%s" % self.config.input_report())
