"""
Deep Hedging Gym.
-----------------
Training environment for deep hedging.

June 30, 2022
@author: hansbuehler
"""
from collections.abc import Mapping
from cdxbasics.prettydict import PrettyDict
from cdxbasics.util import uniqueHash, fmt_list
from cdxbasics.config import Config, Float, Int
from cdxbasics.logger import Logger
import numpy as np
"""
import importlib as imp
import packages.cdx_tf.cdx_tf.gym as _
imp.reload(_)
import packages.cdx_tf.cdx_tf.clip as _
imp.reload(_)
import packages.cdx_tf.cdx_tf.models as _
imp.reload(_)
import packages.cdx_tf.cdx_tf.monetary_utility as _
imp.reload(_)
"""
from packages.cdx_tf.cdx_tf.util import tf, def_dtype, TF_VERSION, tf_make_dim, tfCast
from packages.cdx_tf.cdx_tf.gym import Gym as cdxGym
from packages.cdx_tf.cdx_tf.models import DenseAgent, RecurrentAgent
from packages.cdx_tf.cdx_tf.clip import SoftClip
from packages.cdx_tf.cdx_tf.monetary_utility import MonetaryUtility
from packages.cdx_tf.cdx_tf.optimizer import create_optimizer

_log = Logger(__file__)

class VanillaDeepHedgingGym(cdxGym):
    """
    Vanilla periodic policy search Deep Hedging engine https://arxiv.org/abs/1802.03042
    Vewrsion 3.0 supports recursive networks, caching and training with a standardized framework which is multi-processor enable
    Hans Buehler, June 2022
    """

    Default_features_init = [ 'price' ]
    Default_features_main = Default_features_init + ['time_left', 'delta'  ]

    CACHE_VERSION = "0.0.1"
    LOSS_NAME     = "loss"

    def __init__(self, config : Config, name : str = "VanillaDeepHedging", dtype : tf.DType = def_dtype, trainable : bool = True ):
        """
        Deep Hedging Gym.
        The design pattern here is that the gym instantiates the agent.
        This is done because the gym will know first the number of instruments.
        An alternative design would be to pass the agent as parameter but then
        validate that it has the correct number of instruments.

        Parameters
        ----------
            config : Config
                Sets up the gym, and instantiates the agent
                Main config sections
                    agent     - will be passed to AgentFactory()
                    objective - will be passed to MonetaryUtility()
                    Print config.usage_report() after calling this object
                    for full help
            name : str
                Name of the object for progress mesages
            dtype : tf.DType
                Type
        """
        cdxGym.__init__(self, config=config, name=name, dtype=dtype, trainable=trainable )
        seed                         = config.tensorflow("seed", 423423423, int, "Set tensor random seed. Leave to None if not desired.")
        self.softclip                = SoftClip( config.environment, dtype=self.dtype )
        self._config                 = config.copy()

        self.config_agent            = config.agent.detach()
        self.config_init_agent       = config.init_agent.detach()
        self.config_objective        = config.objective.detach()
        self.agent                   = None
        self.init_delta              = None
        self.utility                 = None
        self.utility0                = None
        self.num_steps               = None
        config.done()

        if not seed is None:
            tf.random.set_seed( seed )

    # -------------------
    # keras model pattern
    # -------------------

    def build(self, shapes : dict ):
        """ Build the model. See call(). """
        assert self.agent is None, "build() called twice?"
        _log.verify( isinstance(shapes, Mapping), "'shapes' must be a dictionary type. Found type %s", type(shapes ))

        nInst             = int( shapes['market']['hedges'][2] )
        self.num_steps    = int( shapes['market']['hedges'][1] )
        self.agent        = RecurrentAgent( nOutput           = nInst,
                                            config            = self.config_agent,
                                            init_config       = self.config_init_agent,
                                            def_features      = self.Default_features_main,
                                            init_def_features = self.Default_features_init,
                                            init_def_width    = 25,
                                            name="main_agent", dtype=self.dtype, trainable=self.trainable )

        self.utility    = MonetaryUtility( config=self.config_objective, def_width=25, def_features=self.Default_features_init, name="utility",  dtype=self.dtype, trainable=self.trainable )
        self.utility0   = MonetaryUtility( config=self.config_objective, def_width=25, def_features=self.Default_features_init, name="utility0", dtype=self.dtype, trainable=self.trainable )

    def call( self, data : dict, training : bool = False ) -> dict:
        """
        Gym track.
        This function expects specific information in the dictionary data; see below

        Parameters
        ----------
            data : dict
                The data for the gym.
                It takes the following data with M=number of time steps, N=number of hedging instruments.
                First coordinate is number of samples in this batch.
                    market, hedges :            (,M,N) the returns of the hedges, per step, per instrument
                    market, cost:               (,M,N) proportional cost for trading, per step, per instrument
                    market, ubnd_a and lbnd_a : (,M,N) min max action, per step, per instrument
                    market, payoff:             (,M) terminal payoff of the underlying portfolio

                    features, per_step:       (,M,N) list of features per step
                    features, per_sample:     (,M) list of features for each sample

            training : bool, optional
                See tensorflow documentation

        Returns
        -------
            dict:
            This function returns analaytics of the performance of the agent
            on the path as a dictionary. Each is returned per sample
                utility:         (,) primary objective to maximize
                utility0:        (,) objective without hedging
                loss:            (,) -utility-utility0
                payoff:          (,) terminal payoff
                pnl:             (,) mid-price pnl of trading (e.g. ex cost)
                cost:            (,) cost of trading
                gains:           (,) total gains: payoff + pnl - cost
                actions:         (,M,N) actions, per step, per path
                deltas:          (,M,N) deltas, per step, per path
        """
        _log.verify( isinstance(data, Mapping), "'data' must be a dictionary type. Found type %s", type(data ))
        assert not self.agent is None and not self.utility is None, "build() not called"

        # geometry
        # --------
        hedges       = data['market']['hedges']
        hedge_shape  = hedges.shape.as_list()
        _log.verify( len(hedge_shape) == 3, "data['market']['hedges']: expected tensor of dimension 3. Found shape %s", hedge_shape )
        nBatch       = hedge_shape[0]    # is None at first call. Will be batch size thereafter
        nSteps       = hedge_shape[1]
        nInst        = hedge_shape[2]

        # extract market data
        # --------------------
        trading_cost = data['market']['cost']
        ubnd_a       = data['market']['ubnd_a']
        lbnd_a       = data['market']['lbnd_a']
        payoff       = data['market']['payoff']
        payoff       = payoff[:,0] if payoff.shape.as_list() == [nBatch,1] else payoff # handle tf<=2.6
        _log.verify( trading_cost.shape.as_list() == [nBatch, nSteps, nInst], "data['market']['cost']: expected shape %s, found %s", [nBatch, nSteps, nInst], trading_cost.shape.as_list() )
        _log.verify( ubnd_a.shape.as_list() == [nBatch, nSteps, nInst], "data['market']['ubnd_a']: expected shape %s, found %s", [nBatch, nSteps, nInst], ubnd_a.shape.as_list() )
        _log.verify( lbnd_a.shape.as_list() == [nBatch, nSteps, nInst], "data['market']['lbnd_a']: expected shape %s, found %s", [nBatch, nSteps, nInst], lbnd_a.shape.as_list() )
        _log.verify( payoff.shape.as_list() == [nBatch], "data['market']['payoff']: expected shape %s, found %s", [nBatch], payoff.shape.as_list() )

        # features
        # --------
        features_per_step, \
        features_per_path = self._features( data, nSteps )

        # main loop
        # ---------
        # V3.0 now supports
        # - recurrent networks
        # - an initial delta add-on which uses a different network than the agent for every time step.
        #   the reason is that if the payoff is unhedged, initial delta is very different than subsequent actions
        # - Tensorflow compilable loop, e.g. the loop below will not be unrolled when tensorflow compiles the function

        # meaningful features at first time step
        features_time_0 = {}
        features_time_0.update( { f:features_per_path[f] for f in features_per_path } )
        features_time_0.update( { f:features_per_step[f][:,0,:] for f in features_per_step})

        # execute initial delta action and initial state
        t             = 0
        action, state = self.agent.initial_action_and_state( features_time_0, training=training )
        cost          = tf.reduce_sum( tf.math.abs( action ) * trading_cost[:,t,:], axis=1, name="cost" )           # [?,]
        pnl           = tf.reduce_sum( action * hedges[:,t,:], axis=1, name="pnl" )                                 # [?,]
        delta         = action
        actions       = tf.stop_gradient(action)[:,tf.newaxis,:]                                                    # [?,0,nInst]

        # execute remaining steps
        t       = 1
        while tf.less(t,nSteps, name="main_loop"): # logically equivalent to: for t in range(nSteps):
            tf.autograph.experimental.set_loop_options( shape_invariants=[(actions, tf.TensorShape([None,None,nInst]))] )

            # 1: build features
            live_features = dict( action=action, delta=delta, cost=cost, pnl=pnl )
            live_features.update( { f:features_per_path[f] for f in features_per_path } )
            live_features.update( { f:features_per_step[f][:,t,:] for f in features_per_step})

            # 2: action
            action, state  =  self.agent( live_features, state=state, training=training )
            _log.verify( action.shape.as_list() == [nBatch, nInst], "Error: action return by agent: expected shape %s, found %s", [nBatch, nInst], action.shape.as_list() )
            action         =  self.softclip( action, lbnd_a[:,t,:], ubnd_a[:,t,:] )
            delta          += action

            # 3: trade
            cost           += tf.reduce_sum( tf.math.abs( action ) * trading_cost[:,t,:], axis=1, name="cost_t" )
            pnl            += tf.reduce_sum( action * hedges[:,t,:], axis=1, name="pnl_t" )

            # 4: record actions per path, per step, continue loop
            actions        =  tf.concat( [ actions,tf.stop_gradient( action )[:,tf.newaxis,:] ], axis=1, name="actions")
            t              += 1  # loop

            action  = tf.debugging.check_numerics(action, "Numerical error computing action in %s. Turn on tf.enable_check_numerics to find the root cause. Note that they are disabled by default in trainer.py" % __file__ )
            pnl     = tf.debugging.check_numerics(pnl, "Numerical error computing pnl in %s. Turn on tf.enable_check_numerics to find the root cause. Note that they are disabled by default in trainer.py" % __file__ )
            cost    = tf.debugging.check_numerics(cost, "Numerical error computing cost in %s. Turn on tf.enable_check_numerics to find the root cause. Note that they are disabled by default in trainer.py" % __file__ )

        # compute utility
        # ---------------

        utility           = self.utility(  X=payoff + pnl - cost, features=features_time_0,  training=training )
        utility0          = self.utility0( X=payoff,              features=features_time_0,  training=training )

        # prepare output
        # --------------

        return PrettyDict(
            loss     = -utility-utility0,                         # [?,] this key must equal self.LOSS_NAME
            utility  = tf.stop_gradient( utility ),               # [?,]
            utility0 = tf.stop_gradient( utility0 ),              # [?,]
            gains    = tf.stop_gradient( payoff + pnl - cost ),   # [?,]
            payoff   = tf.stop_gradient( payoff ),                # [?,]
            pnl      = tf.stop_gradient( pnl ),                   # [?,]
            cost     = tf.stop_gradient( cost ),                  # [?,]
            actions  = tf.stop_gradient( actions )                # [?,nSteps,nInst]
        )

    # ----------------------------------------------------------------
    # Utility function to allow calling the underlying agents
    # ----------------------------------------------------------------

    def extract_initial_features( self, data : dict ) -> dict:
        """ Extract initial features """
        nSteps            = int( data['market']['hedges'].shape[1] )
        features_per_step, \
        features_per_path = self._features( data, nSteps )

        features_time_0 = {}
        features_time_0.update( { f:features_per_path[f] for f in features_per_path } )
        features_time_0.update( { f:features_per_step[f][:,0,:] for f in features_per_step})

        return features_time_0

    # -------------------
    # internal
    # -------------------

    @staticmethod
    def _features( data : dict, nSteps : int ) -> (dict, dict):
        """
        Collect requested features and convert them into common shapes.

        Parameters
        ----------
            data: essentially world.tf_data
            nSteps: for validation.

        Returns
        -------
            features_per_step, features_per_path : (dict, dict)
                features_per_step: requested features which are available per step. Each feature has dimension [nSamples,nSteps,M] for some M
                features_per_path: requested features with dimensions [nSamples,M]
        """
        features             = data.get('features',{})
        features_per_step_i  = features.get('per_step', {})
        features_per_step    = {}
        for f in features_per_step_i:
            feature = features_per_step_i[f]
            assert isinstance(feature, tf.Tensor), "Internal error: type %s found" % type(feature).__name__
            _log.verify( len(feature.shape) >= 2, "data['features']['per_step']['%s']: expected tensor of at least dimension 2, found shape %s", f, feature.shape.as_list() )
            _log.verify( feature.shape[1] == nSteps, "data['features']['per_step']['%s']: second dimension must match number of steps, %ld, found shape %s", f, nSteps, feature.shape.as_list() )
            features_per_step[f] = tf_make_dim( feature, 3 )

        features_per_path_i    = features.get('per_path', {})
        features_per_path      = {}
        assert isinstance( features_per_path_i, dict), "Internal error: type %s found" % type(features_per_path_i).__name__
        for f in features_per_path_i:
            feature = features_per_path_i[f]
            assert isinstance(feature, tf.Tensor), "Internal error: type %s found" % type(feature).__name__
            features_per_path[f] = tf_make_dim( feature, target_dim=2 )
        return features_per_step, features_per_path

    # -------------------
    # syntatic sugar
    # -------------------

    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        assert not self.agent is None, "build() must be called first"
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] )

    def get_config( self ): #NOQA
        return dict( config=self._config,
                     name=self.name,
                     dtype=self.dtype,
                     trainable=self.trainable )
    @staticmethod
    def from_config( self, config ): #NOQA
        return VanillaDeepHedgingGym(**config)

    @property
    def description(self) -> str:
        """ Returns a text description of 'self' """
        text   = "'%(name)s is using a total of %(total)ld weights: %(main)ld for the main agent, %(init_agent)ld for %(recurrence)ld initial states and delta, and %(util)ld weights for the two %(mutil_name)s monetary utilities" % \
                 dict( name=self.name,
                       total=self.num_trainable_weights,
                       main=self.agent.main_agent.num_trainable_weights,
                       init_agent=self.agent.init_agent.num_trainable_weights,
                       recurrence=self.agent.recurrence,
                       mutil_name=self.utility.display_name,
                       util=self.utility.num_trainable_weights+self.utility0.num_trainable_weights)

        text  += "\n"\
                 " Features available for the agent per time step:   %(avl_agent)s\n"\
                 " Features used by the agent per time step:         %(use_agent)s\n"\
                 " Features available for the initial agent:         %(avl_init)s\n"\
                 " Features used by the initial agent:               %(use_init)s\n"\
                 " Features available for the monetary utilities:    %(avl_mutil)s\n"\
                 " Features used by the monetary utilities:          %(use_mutil)s\n" % \
                 dict( avl_agent = fmt_list( self.agent.main_agent.available_features ),
                       use_agent = fmt_list( self.agent.main_agent.features ),
                       avl_init  = fmt_list( self.agent.init_agent.available_features ),
                       use_init  = fmt_list( self.agent.init_agent.features ),
                       avl_mutil = fmt_list( self.utility.available_features ),
                       use_mutil = fmt_list( self.utility.features ) )

        text += "Number of time steps: %ld" % self.num_steps
        return text


