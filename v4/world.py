"""
Deep Hedging Example Worlds
---------------------------
Example world for deep hedging.

June 30, 2022
@author: hansbuehler
"""

from cdxbasics.config import Config
from cdxbasics.prettydict import PrettyDict as pdct
from cdxbasics.logger import Logger
from cdxbasics.dynaplot import figure, colors_tableau
from cdxbasics.util import uniqueHash
from cdxbasics.np import assert_iter_not_is_nan
from packages.cdx_tf.cdx_tf.util import tf, tfCast, tf_dict, def_dtype
from collections.abc import Mapping
import numpy as np
import math as math
#from tqdm import tqdm
from scipy.stats import norm
_log = Logger(__file__)

class SimpleWorld_Spot_ATM(object):
    """
    Simple World with one asset and one floating ATM option.
    The asset has stochastic volatility, and a mean-reverting drift.
    The implied volatility of the asset is not the realized volatility.

    * To use black & scholes mode use hard overwrite black_scholes = True
    * To turn off stochastic vol use no_stoch_vol = True
    * To turn off mean reverrsion of the drift set no_stoch_drift = True

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
            Dictionary of details, e.g. the hidden drift and realized vol of the asset (numpy)
            Most important usually is 'spot_all' which are all spots of the equity (not options), including at maturity (e.g. it has nSteps+1)

        nSamples : int
            Number of samples

        nSteps : int
            Number of steps

        nInst : int
            Number of instruments.

        dt : floast
            Time step.
            TODO: remove in favour of the timeline below

        timelime : np.ndarray
            Generalized timeline. Includes last time point T, e.g. is of length nSteps+1

        config : Config
            Copy of the config file, for cloning

        unique_id : str
            Unique ID generate off the config file, for serialization
    """

    def __init__(self, config : Config, dtype=def_dtype ):
        """
        Parameters
        ----------
        config : Config
            Long list. Use the report feature of 'config' for full feature set

                config  = Config()
                world   = SimpleWorld_Spot_ATM(config)
                print( config.usage_report( with_values = False )

             To use black & scholes mode use hard overwrite black_scholes = True
        """
        self.tf_dtype  = dtype
        self.np_dtype  = dtype.as_numpy_dtype()
        self.unique_id = None               # for serialization; see below
        self.config    = config.copy()      # for cloning

        # simulator
        # ---------
        nSteps     = config("steps", 10, int, help="Number of time steps")
        nSamples   = config("samples", 1000, int, help="Number of samples")
        seed       = config("seed", 2312414312, int, help="Random seed")
        nIvSteps   = config("invar_steps", 5, int, help="Number of steps ahead to sample from invariant distribution")
        dt         = config("dt", 1./50., float, help="Time per timestep.", help_default="One week (1/50)")
        cost_s     = config("cost_s", 0.0002, float, help="Trading cost spot")
        ubnd_as    = config("ubnd_as", 5., float, help="Upper bound for the number of shares traded at each time step")
        lbnd_as    = config("lbnd_as", -5., float, help="Lower bound for the number of shares traded at each time step")
        bs_mode    = config("black_scholes", False, bool, help="Hard overwrite to use a black & scholes model with vol 'rvol' and drift 'drift'. Also turns off the option as a tradable instrument by setting strike = 0.")
        no_svol    = config("no_stoch_vol", False, bool, help="If true, turns off stochastic realized and implied vol, by setting meanrev_*vol = 0 and volvol_*vol = 0")
        no_sdrift  = config("no_stoch_drift", False, bool, help="If true, turns off the stochastic drift of the asset, by setting meanrev_drift = 0. and drift_vol = 0")
        _log.verify( nSteps > 0,    "'steps' must be positive; found %ld", nSteps )
        _log.verify( nSamples > 0,  "'samples' must be positive; found %ld", nSamples )
        _log.verify( dt > 0., "dt must be positive; found %g", dt )
        _log.verify( cost_s >= 0, "'cost_s' must not be negative; found %g", cost_s )
        _log.verify( ubnd_as >= 0., "'ubnd_as' must not be negative; found %g", ubnd_as )
        _log.verify( lbnd_as <= 0., "'lbnd_as' must not be positive; found %g", lbnd_as )
        _log.verify( ubnd_as - lbnd_as > 0., "'ubnd_as - lbnd_as' must be positive; found %g", ubnd_as - lbnd_as)

        # hedging option
        # set strike == 0 to turn off
        strike    = config("strike", 1.,     float, help="Relative strike. Set to zero to turn off option")
        ttm_steps = config("ttm_steps", 4,   int, help="Time to maturity of the option; in steps")
        cost_v    = config("cost_v", 0.02, float, help="Trading cost vega")
        cost_p    = config("cost_p", 0.0005, float, help="Trading cost for the option on top of delta and vega cost")
        ubnd_av   = config("ubnd_av", 5., float, help="Upper bound for the number of options traded at each time step")
        lbnd_av   = config("lbnd_av", -5., float, help="Lower bound for the number of options traded at each time step")
        _log.verify( ttm_steps > 0, "'ttm_steps' must be positive; found %ld", ttm_steps )
        _log.verify( strike >= 0., "'strike' cannot be negative; found %g", strike )
        _log.verify( cost_v >= 0, "'cost_v' must not be negative; found %g", cost_v )
        _log.verify( cost_p >= 0, "'cost_p' must not be negative; found %g", cost_p )
        _log.verify( ubnd_av >= 0., "'ubnd_as' must not be negative; found %g", ubnd_av )
        _log.verify( lbnd_av <= 0., "'lbnd_av' must not be positive; found %g", lbnd_av )
        _log.verify( ubnd_av - lbnd_av > 0., "'ubnd_av - lbnd_as' must be positive; found %g", ubnd_av - lbnd_av )

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

        # market dynamics
        # ---------------
        drift    = config("drift", 0.1, float, help="Mean drift of the asset. This is the total drift.")
        kappa_m  = config("meanrev_drift", 1., float, help="Mean reversion of the drift of the asset")
        xi_m     = config("drift_vol", 0.1, float, help="Vol of the drift")

        # vols
        rvol_init = config("rvol", 0.2, float, help="Initial realized volatility")
        ivol_init = config("ivol", rvol_init, float, help="Initial implied volatility", help_default="Same as realized vol")
        kappa_v   = config("meanrev_rvol", 2., float, help="Mean reversion for realized vol vs implied vol")
        kappa_i   = config("meanrev_ivol", 0.1, float, help="Mean reversion for implied vol vol vs initial level")
        xi_v      = config("volvol_rvol", 0.5, float, help="Vol of Vol for realized vol")
        xi_i      = config("volvol_ivol", 0.5, float, help="Vol of Vol for implied vol")
        _log.verify( rvol_init > 0., "'rvol' must be positive; found %", rvol_init )
        _log.verify( ivol_init > 0., "'ivol' must be positive; found %", ivol_init )

        # correlation
        rho_ms   = config("corr_ms",   0.5, float, help="Correlation between the asset and its mean" )
        rho_vs   = config("corr_vs",  -0.7, float, help="Correlation between the asset and its volatility" )
        rho_vi   = config("corr_vi",   0.8, float, help="Correlation between the implied vol and the asset volatility" )
        rho_vs_r = config("rcorr_vs", -0.5, float, help="Residual correlation between the asset and its implied volatility" )

        _log.verify( abs(rho_ms) <= 1., "'rho_ms' must be between -1 and +1. Found %g", rho_ms )
        _log.verify( abs(rho_vs) <= 1., "'rho_vs' must be between -1 and +1. Found %g", rho_vs )
        _log.verify( abs(rho_vi) <= 1., "'rho_vi' must be between -1 and +1. Found %g", rho_vi )
        _log.verify( abs(rho_vs_r) <= 1., "'rho_vs_r' must be between -1 and +1. Found %g", rho_vs_r )

        # close config
        config.done()
        self.usage_report = config.usage_report()
        self.input_report = config.input_report()

        # black scholes
        if bs_mode:
            strike    = 0.   # turn off option
            ttm_steps = 1
            nIvSteps  = 0
            no_sdrift = True
            no_svol   = True
        if no_sdrift:
            kappa_m   = 0.
            xi_m      = 0.
        if no_svol:
            kappa_v   = 0.
            kappa_i   = 0.
            xi_v      = 0.
            xi_i      = 0.

        # pre compute
        sqrtDt      = math.sqrt(dt)
        ttm_steps   = ttm_steps if strike > 0. else 1
        ttm         = ttm_steps * dt
        sqrtTTM     = math.sqrt(ttm)
        xi_m        = abs(xi_m)   # negative number is odd, but forgivable
        xi_v        = abs(xi_v)   # negative number is odd, but forgivable
        xi_i        = abs(xi_i)   # negative number is odd, but forgivable
        time_left   = np.linspace(float(nSteps), 1., nSteps, endpoint=True, dtype=self.np_dtype) * dt
        sqrt_time_left = np.sqrt( time_left )

        # simulate
        # --------
        # Not the most efficient simulator, but easier to read this way

        np.random.seed(seed)
        dW          = np.random.normal(size=(nSamples,nSteps+nIvSteps+ttm_steps-1,4)).astype( self.np_dtype ) * sqrtDt
        dW_s        = dW[:,:,0]
        dW_m        = dW[:,:,0] * rho_ms + math.sqrt(1. - rho_ms**2) * dW[:,:,1]
        dW_v        = dW[:,:,0] * rho_vs + math.sqrt(1. - rho_vs**2) * dW[:,:,2]
        dW_i        = dW[:,:,2] * rho_vi + math.sqrt(1. - rho_vi**2) * ( dW[:,:,0] * rho_vs_r + math.sqrt(1. - rho_vs_r**2) * dW[:,:,3] )

        spot            = np.zeros((nSamples,nSteps+nIvSteps+ttm_steps), dtype=self.np_dtype)
        rdrift          = np.full((nSamples,nSteps+nIvSteps+ttm_steps), drift, dtype=self.np_dtype)
        rvol            = np.full((nSamples,nSteps+nIvSteps+ttm_steps), rvol_init, dtype=self.np_dtype)
        ivol            = np.full((nSamples,nSteps+nIvSteps+ttm_steps), ivol_init, dtype=self.np_dtype)

        spot[:,0]       = 1.
        log_ivol_init   = np.log( ivol_init )
        log_rvol        = np.log( rvol_init )
        log_ivol        = log_ivol_init
        rvol[:,0]       = rvol_init
        ivol[:,0]       = ivol_init
        mrdrift         = 0. * dW_m[:,0]
        expdriftdt      = np.exp( drift * dt )
        bStochDrift     = kappa_m != 0. or xi_m != 0.
        bStochVol       = kappa_v != 0. or xi_v != 0.  or kappa_i != 0. or xi_i != 0.

        for j in range(1,nSteps+nIvSteps+ttm_steps):
            # spot
            spot[:,j]      = spot[:,j-1] * np.exp( rdrift[:,j-1] * dt + rvol[:,j-1] * dW_s[:,j-1] - 0.5 * (rvol[:,j-1] ** 2) * dt )
            spot[:,j]      *= expdriftdt / np.mean( spot[:,j] )

            # drift
            # we normalize the stochastic drift to 'drift' on average.
            if bStochDrift:
                mrdrift        = mrdrift - kappa_m * mrdrift * dt + xi_m * dW_m[:,j-1]
                mrdrift        = np.exp( mrdrift * dt )
                mrdrift        = np.log( mrdrift / np.mean( mrdrift ) ) / dt
                rdrift[:,j]    = drift + mrdrift

            # vols
            if bStochVol:
                log_rvol       += kappa_v * ( log_ivol      - log_rvol ) * dt + xi_v * dW_v[:,j-1] - 0.5 * (xi_v ** 2) * dt
                log_ivol       += kappa_i * ( log_ivol_init - log_ivol ) * dt + xi_i * dW_i[:,j-1] - 0.5 * (xi_i ** 2) * dt
                rvol[:,j]      = np.exp( log_rvol )
                ivol[:,j]      = np.exp( log_ivol )

        # throw away the first nInvSteps
        # so we start in an invariant distribution
        spot       = spot[:,nIvSteps:]
        rdrift     = rdrift[:,nIvSteps:nIvSteps+nSteps]
        rvol       = rvol[:,nIvSteps:nIvSteps+nSteps]
        ivol       = ivol[:,nIvSteps:nIvSteps+nSteps]

        # sort
        ixs        = np.argsort( spot[:,nSteps] )
        spot       = spot[ixs,:]
        rdrift     = rdrift[ixs,:]
        rvol       = rvol[ixs,:]
        ivol       = ivol[ixs,:]

        # hedging instruments
        # -------------------

        dS         = spot[:,nSteps][:,np.newaxis] - spot[:,:nSteps]
        cost_dS    = spot[:,:nSteps] * cost_s
        if strike <= 0.:
            dInsts     = dS[:,:,np.newaxis]
            cost       = cost_dS[:,:,np.newaxis]
            price      = spot[:,:nSteps]
            ubnd_a     = np.full( (nSamples,nSteps,1), ubnd_as )
            lbnd_a     = np.full( (nSamples,nSteps,1), lbnd_as )

            call_price = None
            call_delta = None
            call_vega  = None
            cost_dC    = None

        else:
            # add hedging instrument: calls
            mat_spot   = spot[:,ttm_steps:ttm_steps+nSteps]   # spot at maturity of each option
            opt_spot   = spot[:,:nSteps]                      # spot at trading date of each option
            payoffs    = np.maximum( 0, mat_spot - strike * opt_spot )
            d1         = ( - np.log( strike ) + 0.5 * ivol * ivol * ttm) / ( ivol * sqrtTTM )
            d2         = d1 - ivol * sqrtTTM
            N1         = norm.cdf(d1)
            N2         = norm.cdf(d2)
            call_price = N1 * opt_spot - N2 * strike * opt_spot
            dC         = payoffs - call_price
            call_delta = N1
            call_vega  = opt_spot * norm.pdf(d1) * sqrtTTM
            cost_dC    = cost_v * np.abs(call_vega) + cost_s * np.abs(call_delta) + cost_p * abs(call_price) # note: for a call vega and delta are positive, but we apply abs() anyway to illusteate the point

            dInsts         = np.ones((nSamples,nSteps,2), dtype=self.np_dtype)
            cost           = np.ones((nSamples,nSteps,2), dtype=self.np_dtype)
            price          = np.ones((nSamples,nSteps,2), dtype=self.np_dtype)
            ubnd_a         = np.ones((nSamples,nSteps,2), dtype=self.np_dtype)
            lbnd_a         = np.ones((nSamples,nSteps,2), dtype=self.np_dtype)
            dInsts[:,:,0]  = dS
            dInsts[:,:,1]  = dC
            cost[:,:,0]    = cost_dS
            cost[:,:,1]    = cost_dC
            price[:,:,0]   = spot[:,:nSteps]
            price[:,:,1]   = call_price
            ubnd_a[:,:,0]  = ubnd_as
            ubnd_a[:,:,1]  = ubnd_av
            lbnd_a[:,:,0]  = lbnd_as
            lbnd_a[:,:,1]  = lbnd_av

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
        # unique_id
        # -----------------------------
        # Default handling for configs will ignore any function definitions, e.g. in this case 'payoff'.
        # we therefore manually generate a sufficient hash
        self.unique_id = uniqueHash( [ config.input_dict(), payoff_f, self.tf_dtype.name ],parse_functions = True )

        # -----------------------------
        # store data
        # -----------------------------

        # market
        # note that market variables are *not* automatically features
        # as trhey often look ahead

        self.data = pdct()
        self.data.market = pdct(
                hedges    = dInsts,
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
                cost   = cost,            # trading cost
                price  = price,           # price
                ubnd_a = ubnd_a,          # bounds. Currently those are determinstic so don't use as features
                lbnd_a = lbnd_a,
                                          # time
                time_left      = np.full( (nSamples, nSteps), time_left[np.newaxis,:], dtype=self.np_dtype ),
                sqrt_time_left = np.full( (nSamples, nSteps), sqrt_time_left[np.newaxis,:], dtype=self.np_dtype ),
                # specific to equity spot
                spot   = spot[:,:nSteps], # spot level (S0,....,Sm-1). This does not include the terminal spot level.
                ivol   = ivol,            # implied vol at beginning of each interval
                ),
            per_path = pdct(),
            )
        if strike > 0.:
            self.data.features.per_step.update(
                call_price  = call_price,          # price of the option
                call_delta  = call_delta,          # delta
                call_vega   = call_vega,           # vega
                cost_v      = cost_dC
            )

        if not py_feat is None:
            self.data.features.per_step.payoff_features = py_feat

        # check numerics
        assert_iter_not_is_nan( self.data, "data" )

        # data
        # what gym() gets

        self.tf_data = tf_dict(
            features = self.data.features,
            market   = self.data.market,
            dtype    = self.tf_dtype
            )

        # details
        # variables for visualization, but not available for the agent
        self.details = pdct(
            # mandatory (used by ploting)
            spot_all     = spot[:,:nSteps+1],      # [nSamples,nSteps+1] spots including spot at T
            # per model
            drift     = rdrift,   # drifts for each interval
            rvol      = rvol,     # realized vols for each interval
            )

        # check numerics
        assert_iter_not_is_nan( self.details, "details" )

        # generating sample weights
        # the tf_sample_weights is passed to keras train and must be of size [nSamples,1]
        # https://stackoverflow.com/questions/60399983/how-to-create-and-use-weighted-metrics-in-keras
        self.sample_weights = np.full((nSamples,1),1./float(nSamples), dtype=self.np_dtype)
        self.tf_sample_weights \
                       = tf.constant( self.sample_weights, dtype=self.tf_dtype)   # must be of size [nSamples,1] https://stackoverflow.com/questions/60399983/how-to-create-and-use-weighted-metrics-in-keras
        self.sample_weights = self.sample_weights.reshape((nSamples,))
        self.tf_y      = tf.zeros((nSamples,), dtype=self.tf_dtype)
        self.nSteps    = nSteps
        self.nSamples  = nSamples
        self.nInst     = 1 if strike <= 0. else 2
        self.dt        = dt
        self.timeline  = np.cumsum( np.linspace( 0., nSteps, nSteps+1, endpoint=True, dtype=np.float32 ) ) * dt

        self.inst_names = [ 'spot' ]
        if strike > 0.:
            self.inst_names.append( "ATM Call" )

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
        return SimpleWorld_Spot_ATM( config )

    def plot(self, config = Config(), **kwargs ):
        """ Plot simple world """

        config.update(kwargs)
        col_size     = config.fig("col_size", 5, int, "Figure column size")
        row_size     = config.fig("row_size", 5, int, "Figure row size")
        plot_samples = config("plot_samples", 5, int, "Number of samples to plot")
        print_input  = config("print_input", True, bool, "Whether to print the config inputs for the world")

        xSamples  = np.linspace(0,self.nSamples,plot_samples,endpoint=False, dtype=int)
        timeline1 = np.cumsum( np.linspace( 0., self.nSteps, self.nSteps+1, endpoint=True, dtype=np.float32 ) ) * self.dt
        timeline  = timeline1[:-1]

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
#        ax.get_xaxis().get_major_formatter().get_useOffset(False)
        ax.legend()

        # drift
        ax  = fig.add_plot()
        ax.set_title("Drift")
        ax.set_xlabel("Time")
        for i, color in zip( xSamples, colors_tableau() ):
            ax.plot( timeline, self.details.drift[i,:], "-", color=color )
        ax.plot( timeline, np.mean( self.details.drift, axis=0), "_", color="black", label="mean" )

        # vols
        ax  = fig.add_plot()
        ax.set_title("Volatilities")
        ax.set_xlabel("Time")
        for i, color in zip( xSamples, colors_tableau() ):
            ax.plot( timeline, self.data.features.per_step.ivol[i,:], "-", color=color )
            ax.plot( timeline, self.details.rvol[i,:], ":", color=color )

        if self.nInst > 1:
            # call prices
            ax  = fig.add_plot(True)
            ax.set_title("Call Prices")
            ax.set_xlabel("Time")
            for i, color in zip( xSamples, colors_tableau() ):
                ax.plot( timeline, self.data.features.per_step.call_price[i,:], "-", color=color )
            ax.plot( timeline, np.mean( self.data.features.per_step.call_price, axis=0), "_", color="black", label="mean" )
            ax.legend()

            # call delta
            ax  = fig.add_plot()
            ax.set_title("Call Deltas")
            ax.set_xlabel("Time")
            for i, color in zip( xSamples, colors_tableau() ):
                ax.plot( timeline, self.data.features.per_step.call_delta[i,:], "-", color=color )
            ax.plot( timeline, np.mean( self.data.features.per_step.call_delta, axis=0), "_", color="black", label="mean" )
            ax.legend()

            # call vega
            ax  = fig.add_plot()
            ax.set_title("Call Vegas")
            ax.set_xlabel("Time")
            for i, color in zip( xSamples, colors_tableau() ):
                ax.plot( timeline, self.data.features.per_step.call_vega[i,:], "-", color=color )
            ax.plot( timeline, np.mean( self.data.features.per_step.call_vega, axis=0), "_", color="black", label="mean" )
            ax.legend()

        fig.render()
        del fig

        if print_input:
            print("Config settings:\n%s" % self.input_report)

