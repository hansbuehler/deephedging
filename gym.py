"""
Deep Hedging Gym.
-----------------
Training environment for deep hedging.

June 30, 2022
@author: hansbuehler
"""
from .base import Logger, Config, tf, dh_dtype, pdct, tf_make_dim, Int, Float, tfCast, create_optimizer, TF_VERSION
from .agents import AgentFactory
from .objectives import MonetaryUtility
from .softclip import DHSoftClip
from collections.abc import Mapping
from cdxbasics.util import uniqueHash
import numpy as np

_log = Logger(__file__)

class VanillaDeepHedgingGym(tf.keras.Model):
    """
    Vanilla periodic policy search Deep Hedging engine https://arxiv.org/abs/1802.03042
    Vewrsion 2.0 supports recursive and iterative networks
    Hans Buehler, June 2022
    """

    def __init__(self, config : Config, name : str = "VanillaDeepHedging", dtype = dh_dtype ):
        """
        Deep Hedging Gym.
        The design pattern here is that the gym instantiates the agent.
        This is done because the gym will know first the number of instruemnt.
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
        tf.keras.Model.__init__(self, name=name, dtype=dtype )
        seed                       = config.tensorflow("seed", 423423423, int, "Set tensor random seed. Leave to None if not desired.")
        self.softclip              = DHSoftClip( config.environment )
        self.config_agent          = config.agent.detach()
        self.config_objective      = config.objective.detach()
        self.user_version          = config("user_version", None, help="An arbitrary string which can be used to identify a particular gym. Changing this value will generate a new cache key")
        self.agent                 = None
        self.utility               = None
        self.utility0              = None
        self.unique_id             = config.unique_id()  # for serialization
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

        nInst         = int( shapes['market']['hedges'][2] )
        self.agent    = AgentFactory( nInst, self.config_agent, name="agent",    dtype=self.dtype )
        self.utility  = MonetaryUtility( self.config_objective, name="utility",  dtype=self.dtype )
        self.utility0 = MonetaryUtility( self.config_objective, name="utility0", dtype=self.dtype )

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
        return self._call( tfCast(data), training )
    @tf.function
    def _call( self, data : dict, training : bool ) -> dict:
        """ The _call function was introduced to allow conversion of numpy arrays into tensors ahead of tf.function tracing """
        _log.verify( isinstance(data, Mapping), "'data' must be a dictionary type. Found type %s", type(data ))
        assert not self.agent is None and not self.utility is None, "build() not called"

        # geometry
        # --------
        hedges       = data['market']['hedges']
        hedge_shape  = hedges.shape.as_list()
        _log.verify( len(hedge_shape) == 3, "data['market']['hedges']: expected tensor of dimension 3. Found shape %s", hedge_shape )
        nBatch       = hedge_shape[0]    # is None at first call. Later will be batch size
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
        # V2.0 now supports
        # - recurrent networks
        # - an initial delta add-on which uses a different network than the agent for every time step.
        #   the reason is that if the payoff is unhedged, initial delta is very different than subsequent actions
        # - Tensorflow compilable loop, e.g. the loop below will not be unrolled when tensorflow compiles the function

        # meaningful features at first time step
        features_time_0 = {}
        features_time_0.update( { f:features_per_path[f] for f in features_per_path } )
        features_time_0.update( { f:features_per_step[f][:,0,:] for f in features_per_step})

        # initialize variable and obtain initial recurrent state, if any
        pnl     = tf.zeros_like(payoff, dtype=dh_dtype)                                                       # [?,]
        cost    = tf.zeros_like(payoff, dtype=dh_dtype)                                                       # [?,]
        delta   = tf.zeros_like(trading_cost[:,0,:], dtype=dh_dtype)                                          # [?,nInst]
        action  = tf.zeros_like(trading_cost[:,0,:], dtype=dh_dtype)                                          # [?,nInst]
        actions = tf.zeros_like(trading_cost[:,0,:][:,tf.newaxis,:], dtype=dh_dtype)                          # [?,0,nInst]
        state   = self.agent.initial_state( features_time_0, training=training ) if self.agent.is_recurrent else tf.zeros_like(pnl, dtype=dh_dtype)  # [?,nStates] if states are used else [?]
        idelta  = self.agent.initial_delta( features_time_0, training=training ) if self.agent.has_initial_delta else tf.zeros_like(delta, dtype=dh_dtype)  # [?,nInst]

        t       = 0
        while tf.less(t,nSteps, name="main_loop"): # logically equivalent to: for t in range(nSteps):
            tf.autograph.experimental.set_loop_options( shape_invariants=[(actions, tf.TensorShape([None,None,nInst]))] )

            # 1: build features, including recurrent state
            live_features = dict( action=action, delta=delta, cost=cost, pnl=pnl )
            live_features.update( { f:features_per_path[f] for f in features_per_path } )
            live_features.update( { f:features_per_step[f][:,t,:] for f in features_per_step})
            if self.agent.is_recurrent: live_features[ self.agent.state_feature_name ] = state

            # 2: action
            action, state_ =  self.agent( live_features, training=training )
            _log.verify( action.shape.as_list() == [nBatch, nInst], "Error: action return by agent: expected shape %s, found %s", [nBatch, nInst], action.shape.as_list() )
            action         += idelta
            action         =  self.softclip(action, lbnd_a[:,t,:], ubnd_a[:,t,:] )
            state          =  state_ if self.agent.is_recurrent else state
            delta          += action

            # 3: trade
            cost           += tf.reduce_sum( tf.math.abs( action ) * trading_cost[:,t,:], axis=1, name="cost_t" )
            pnl            += tf.reduce_sum( action * hedges[:,t,:], axis=1, name="pnl_t" )

            # 4: record actions per path, per step, continue loop
            action_        =  tf.stop_gradient( action )[:,tf.newaxis,:]
            actions        =  tf.concat( [actions,action_], axis=1, name="actions") if t>0 else action_
            idelta         *= 0. # no more initial delta
            t              += 1  # loop

        pnl  = tf.debugging.check_numerics(pnl, "Numerical error computing pnl in %s. Turn on tf.enable_check_numerics to find the root cause. Note that they are disabled by default in trainer.py" % __file__ )
        cost = tf.debugging.check_numerics(cost, "Numerical error computing cost in %s. Turn on tf.enable_check_numerics to find the root cause. Note that they are disabled by default in trainer.py" % __file__ )

        # compute utility
        # ---------------

        utility           = self.utility( data=dict(features_time_0 = features_time_0,
                                                    payoff          = payoff,
                                                    pnl             = pnl,
                                                    cost            = cost ), training=training )
        utility0          = self.utility0(data=dict(features_time_0 = features_time_0,
                                                    payoff          = payoff,
                                                    pnl             = pnl*0.,
                                                    cost            = cost*0.), training=training )
        # prepare output
        # --------------

        return pdct(
            loss     = -utility-utility0,                         # [?,]
            utility  = tf.stop_gradient( utility ),               # [?,]
            utility0 = tf.stop_gradient( utility0 ),              # [?,]
            gains    = tf.stop_gradient( payoff + pnl - cost ),   # [?,]
            payoff   = tf.stop_gradient( payoff ),                # [?,]
            pnl      = tf.stop_gradient( pnl ),                   # [?,]
            cost     = tf.stop_gradient( cost ),                  # [?,]
            actions  = tf.concat( actions, axis=1, name="actions" ) # [?,nSteps,nInst]
        )

    # -------------------
    # internal
    # -------------------

    @staticmethod
    def _features( data : dict, nSteps : int = None ) -> (dict, dict):
        """
        Collect requested features and convert them into common shapes.

        Parameters
        ----------
            data: essentially world.tf_data
            nSteps: for validation. Can be left None to ignore.

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
            assert isinstance(feature, tf.Tensor), "Internal error: type %s found" % feature.__class__.__name__
            _log.verify( len(feature.shape) >= 2, "data['features']['per_step']['%s']: expected tensor of at least dimension 2, found shape %s", f, feature.shape.as_list() )
            if not nSteps is None: _log.verify( feature.shape[1] == nSteps, "data['features']['per_step']['%s']: second dimension must match number of steps, %ld, found shape %s", f, nSteps, feature.shape.as_list() )
            features_per_step[f] = tf_make_dim( feature, 3 )

        features_per_path_i    = features.get('per_path', {})
        features_per_path      = {}
        assert isinstance( features_per_path_i, dict), "Internal error: type %s found" % features_per_path_i.__class__.__name__
        for f in features_per_path_i:
            feature = features_per_path_i[f]
            assert isinstance(feature, tf.Tensor), "Internal error: type %s found" % feature.__class__.__name__
            features_per_path[f] = tf_make_dim( feature, dim=2 )
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

    @property
    def available_features_per_step(self) -> list:
        """ Returns the list of features available per time step (for the agent). The model must have been call()ed once """
        _log.verify( not self.agent is None, "Cannot call this function before model was built")
        return self.agent.available_features

    @property
    def available_features_per_path(self) -> list:
        """ Returns the list of features available per time step (for montetary utilities). The model must have been call()ed once """
        _log.verify( not self.utility is None, "Cannot call this function before model was built")
        return self.utility.available_features

    @property
    def agent_features_used(self) -> list:
        """ Returns the list of features used by the agent. The model must have been call()ed once """
        _log.verify( not self.agent is None, "Cannot call this function before model was built")
        return self.agent.public_features

    @property
    def utility_features_used(self) -> list:
        """ Returns the list of features available per time step (for the agent). The model must have been call()ed once """
        _log.verify( not self.agent is None, "Cannot call this function before model was built")
        return self.utility.features

    # -------------------
    # caching
    # -------------------

    def create_cache( self ):
        """
        Create a dictionary which allows reconstructing the current model.
        The content of the dictionary are IDs to validate that we are reconstructing the same type of gym,
        weights of the gym and the optimizer, and the last learning rate of the optimizer.

        Note: reconstruction of an optimizer state is not natively supported in TensorFlow. Below might not work perfectly.
        """
        assert not self.agent is None, "build() not called yet"
        opt_weights = self.optimizer.get_weights() if not getattr(self.optimizer,"get_weights",None) is None else None
        opt_config  = tf.keras.optimizers.serialize( self.optimizer )['config'] if not self.optimizer is None else None

        if not opt_config is None and opt_weights is None:
            # tensorflow 2.11 abandons 'get_weights'
            variables   = self.optimizer.variables()
            opt_weights = [ np.array( v ) for v in variables ]

        # we compute a config ID for all parameters but the learning rate
        # That should work for most optimizers, but future optimizers may
        # rquire copying furhter variables
        id_config   = { k: opt_config[k] for k in opt_config if k != 'learning_rate' } if not opt_config is None else None
        opt_uid     = uniqueHash( id_config ) if not id_config is None else ""
        opt_weights = self.optimizer.get_weights() if TF_VERSION <= 210 else [ w.value() for w in self.optimizer.variables() ]

        return dict( gym_uid       = self.unique_id,
                     gym_weights   = self.get_weights(),
                     opt_uid       = opt_uid,
                     opt_config    = opt_config,
                     opt_weights   = opt_weights
                   )

    def restore_from_cache( self, cache ) -> bool:
        """
        Restore 'self' from cache.
        Note that we have to call() this object before being able to use this function
        This function returns False if the cached weights do not match the current architecture.

        Note: reconstruction of an optimizer state is not natively supported in TensorFlow. Below might not work perfectly.
        """
        assert not self.agent is None, "build() not called yet"
        gym_uid     = cache['gym_uid']
        gym_weights = cache['gym_weights']
        opt_uid     = cache['opt_uid']
        opt_config  = cache['opt_config']
        opt_weights = cache['opt_weights']

        self_opt_config = tf.keras.optimizers.serialize( self.optimizer )['config'] if not self.optimizer is None else None
        self_id_config  = { k: opt_config[k] for k in opt_config if k != 'learning_rate' } if not self_opt_config is None else None
        self_opt_uid    = uniqueHash( self_id_config ) if not self_opt_config is None else ""

        # check that the objects correspond to the correct configs
        if gym_uid != self.unique_id:
            _log.warn( "Cache restoration error: provided cache object has gym ID %s vs current ID %s", gym_uid, self.unique_id)
            return False
        if opt_uid != self_opt_uid:
            _log.warn( "Cache restoration error: provided cache object has optimizer ID %s vs current ID %s\n"\
                       "Stored configuration: %s\nCurrent configuration: %s", opt_uid, self_opt_uid, opt_config, self_opt_config)
            return False

        # load weights
        # Note that we will continue with the restored weights for the gym even if we fail to restore the optimizer
        # This is likely the desired behaviour.
        try:
            self.set_weights( gym_weights )
        except ValueError as v:
            _log.warn( "Cache restoration error: provided cache gym weights were not compatible with the gym.\n%s", v)
            return False
        return True

        if self.optimizer is None:
            return True
        # set learning rate to last recoreded value
        if 'learning_rate' in opt_config:
            self.optimizer.learning_rate = opt_config['learning_rate']
        # restore weights
        try:
            self.optimizer.set_weights( opt_weights )
        except ValueError as v:
            isTF211 = getattr(self.optimizer,"get_weights",None) is None
            isTF211 = "" if not isTF211 else "Code is running TensorFlow 2.11 or higher for which tf.keras.optimizers.Optimizer.get_weights() was retired. Current code is experimental. Review create_cache/restore_from_cache.\n"
            _log.warn( "Cache restoration error: cached optimizer weights were not compatible with existing optimizer.\n%s%s", v)
            return False


        return True

