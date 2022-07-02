"""
Deep Hedging Gym.
-----------------
Training environment for deep hedging.

June 30, 2022
@author: hansbuehler
"""
from .base import Logger, Config, tf, tfp, dh_dtype, pdct, tf_back_flatten, tf_make_dim
from .agents import AgentFactory
from .objectives import MonetaryUtility
from collections.abc import Mapping
_log = Logger(__file__)

class VanillaDeepHedgingGym(tf.keras.Model):
    """ 
    Vanilla periodic policy search Deep Hedging engine https://arxiv.org/abs/1802.03042 
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
        tf.keras.Model.__init__(self, dtype=dtype )
        self.config_action         = config.agent.detach()
        self.config_utility        = config.objective.detach()
        hinge_softness             = config.environment('softclip_hinge_softness', 1., float, help="Specifies softness of bounding actions between lbnd_a and ubnd_a")
        self.softclip              = tfp.bijectors.SoftClip( low=0., high=1., hinge_softness=hinge_softness, name='soft_clip' )
        config.done()
        
        self.agent                 = None
        self.utility               = None
        
    def build(self, shapes : dict ):
        """ Build the model. See call(). """
        assert self.agent is None and self.utility is None, "build() called twioce?"
        _log.verify( isinstance(shapes, Mapping), "'shapes' must be a dictionary type. Found type %s", type(shapes ))

        nInst         = int( shapes['market']['hedges'][2] )
        self.agent    = AgentFactory( nInst, self.config_action, dtype=self.dtype ) 
        self.utility  = MonetaryUtility( self.config_utility, dtype=self.dtype ) 
        self.utility0 = MonetaryUtility( self.config_utility, dtype=self.dtype ) 
        
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
                utiliy:          (,) primary objective to maximize
                utiliy0:         (,) objective without hedging
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
        nBatch       = hedge_shape[0]    # is None at first call. Later will be batch size
        nSteps       = hedge_shape[1]
        nInst        = hedge_shape[2]
        
        # extract market data
        # --------------------
        trading_cost = data['market']['cost']
        ubnd_a       = data['market']['ubnd_a']
        lbnd_a       = data['market']['lbnd_a']
        payoff       = data['market']['payoff']
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

        pnl     = tf.zeros_like(payoff)
        cost    = tf.zeros_like(payoff)
        delta   = tf.zeros_like(trading_cost[:,0,:])
        action  = tf.zeros_like(trading_cost[:,0,:])
        actions = []
        deltas  = []        

        with tf.control_dependencies( [ tf.debugging.assert_greater_equal( cost, 0., message="Cost for actions must not be negative" ),
                                        tf.debugging.assert_all_finite( pnl - cost, message="Infinite values found")
                                        ] ):
                
            for j in range(nSteps):
                # build features
                live_features = dict( action=action, delta=delta, cost=cost, pnl=pnl )
                live_features.update( { f:features_per_path[f] for f in features_per_path } )
                live_features.update( { f:features_per_step[f][:,j,:] for f in features_per_step})
                live_features['delta'] = delta
                live_features['action'] = action
    
                # action
                action  =  self.agent( live_features, training=training )
                _log.verify( action.shape.as_list() == [nBatch, nInst], "action: expected shape %s, found %s", [nBatch, nInst], action.shape.as_list() )
                action  =  self._clip_actions(action, lbnd_a[:,j,:], ubnd_a[:,j,:] )
                delta   =  delta + action
                
                # trade
                cost    += tf.reduce_sum( tf.math.abs( action ) * trading_cost[:,j,:], axis=1 )
                pnl     += tf.reduce_sum( action * hedges[:,j,:], axis=1 )
                
                actions.append( tf.stop_gradient( action )[:,tf.newaxis,:] )
                deltas.append( tf.stop_gradient( delta )[:,tf.newaxis,:] )
    
            # compute utility
            # ---------------
    
            features_time_0 = {}
            features_time_0.update( { f:features_per_path[f] for f in features_per_path } )
            features_time_0.update( { f:features_per_step[f][:,0,:] for f in features_per_step})
    
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
                loss     = -utility-utility0,
                utility  = tf.stop_gradient( utility ),
                utility0 = tf.stop_gradient( utility0 ),
                gains    = tf.stop_gradient( payoff + pnl - cost ),
                payoff   = tf.stop_gradient( payoff ),
                pnl      = tf.stop_gradient( pnl ),
                cost     = tf.stop_gradient( cost ),
                actions  = tf.concat( actions, axis=1 ),
                deltas   = tf.concat( deltas, axis=1 )
            )
            
    def _clip_actions( self, actions, lbnd_a, ubnd_a ):
        """ Clip the action within lbnd_a, ubnd_a """
        with tf.control_dependencies( [ tf.debugging.assert_greater_equal( ubnd_a, lbnd_a, message="Upper bound for actions must be bigger than lower bound" ),
                                        tf.debugging.assert_greater_equal( ubnd_a, 0., message="Upper bound for actions must not be negative" ),
                                        tf.debugging.assert_less_equal( lbnd_a, 0., message="Lower bound for actions must not be positive" ) ] ):
            dbnd = ubnd_a - lbnd_a
            rel  = ( actions - lbnd_a ) / dbnd
            rel  = self.softclip( rel )
            return tf.where( dbnd > 0., rel *  dbnd + lbnd_a, 0. )

    def _features( self, data : dict, nSteps : int) -> (dict, dict):
        """ 
        Collect requested features and convert them into common shapes.    
        
        Returns
        -------
            features_per_step, features_per_path : (dict, dict)
                features_per_step: requested features which are available per step. Each feature has dimension [nSamples,nSteps,M] for some M
                features_per_path: requested features with dimensions [nSamples,M]
        """
        features             = data.get('features',{})

        features_per_step_i  = features.get('per_step', {})
        features_per_step    =   {}
        for f in features_per_step_i:
            feature = features_per_step_i[f]
            assert isinstance(feature, tf.Tensor), "Internal error: type %s found" % feature._class__.__name__
            _log.verify( len(feature.shape) >= 2, "data['features']['per_step']['%s']: expected tensor of at least dimension 2, found shape %s", f, feature.shape.as_list() )
            _log.verify( feature.shape[1] == nSteps, "data['features']['per_step']['%s']: second dimnsion must match number of steps, %ld, found shape %s", f, nSteps, feature.shape.as_list() )
            features_per_step[f] = tf_make_dim( feature, 3 )

        features_per_path_i    = features.get('per_sample', {})
        features_per_path      = { tf_make_dim( _, dim=2 ) for _ in features_per_path_i }
        
        return features_per_step, features_per_path
            
