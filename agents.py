# -*- coding: utf-8 -*-
"""
Deep Hedging Agents
-------------------
Contains simple training agents

June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype, Int, fmt_list
from .layers import DenseLayer, VariableLayer
import numpy as np
from collections.abc import Mapping
_log = Logger(__file__)

class SimpleDenseAgent(tf.keras.layers.Layer):
    """
    Simple Action Model for Deep Hedging
    V2.0 also supports simple recurrence in the form of a new feature, which is returned by the previous agent call.
    
    To initialize an agent, you need to specify its network (with layers.DenseLayer) and its features.
    The list of available features for a given world in a given gym can be obtained using gym.available_features_per_step()
    """
    
    Default_features_per_step = [ 'price', 'delta', 'time_left' ]
    State_Feature_Name = "recurrent_state"

    def __init__(self, nInst : int, config : Config, name : str = None, dtype : tf.DType = dh_dtype ):
        """
        Create an agent which returns the action for the given Deep Hedging problem
        See tf.keras.layers.Layer.__call__ for comments
        
        The agent's __call__ function will take in a dictionary of tenors of all feature avaialble,
        and returns the corresponding action for 'nInst' instruments, and any recurrrent states.
        
        See Network.md for a summary of the network definition provided by this file.
        
        Parameters
        ----------
            nInst : int
                Number of instruments
            config : Config
                Configuration for this object. Most notably
                    network : Config
                        network.width
                        network.activation
                        network.depth
                    features : list
                        Set of features by name the network will use
            name : str, optional
                Name of the layer
            per_step : bool, optional
                Whether the agent is used per time step, or once per sample.
            dtype : tf.DType, optional
                dtype
        """
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )
        features                = config("features", self.Default_features_per_step, list, "Named features for the agent to use")
        state_features          = config.state("features", [], list, "Named features for the agent to use for the initial state network")
        init_delta_features     = config.init_delta("features", [], list, "Named features for the agent to use for the initial delta network")
        init_delta              = config.init_delta("active", True, bool, "Whether or not to train in addition a delta layer for the first step")

        self.classic_states     = config.recurrence.states("classic",   0, Int>=0, "Number of 'classic' recurrent states to be used. Such states may suffer from long term memory loss and gradient explosion. Classic states are constrained to (-1,+1)")
        self.aggregate_states   = config.recurrence.states("aggregate", 0, Int>=0, "Number of 'aggregate' states to be used. Such states capture in spirit exponentially weighted characteristics of the path")
        self.past_repr_states   = config.recurrence.states("past_repr", 0, Int>=0, "Number of 'past representation' states to be used. Such states capture data from past dates such as the spot value at the last reset date")
        self.event_states       = config.recurrence.states("event",     0, Int>=0, "Number of 'event' states to be used. Such states capture digital events such as a barrier breach")
        self.bound_aggr_states  = config.recurrence("bound_aggr_states", False, bool, "Whether or not to bound aggregate states to (-1,+1)")
        self.sigmoid_1          = config.recurrence("sigmoid_1", False, bool, "Whether to use sigmoid function for digitial states (experimental)")

        self.nInst              = int(nInst)
        self.nUpdateUnits       = self.aggregate_states + self.past_repr_states + self.event_states 
        self.nStates            = self.classic_states + self.aggregate_states + self.past_repr_states + self.event_states 
        
        _log.verify( self.State_Feature_Name not in features, "Cannot use internal state name '%s' in feature list", self.State_Feature_Name )
        
        is_recurrent            = self.nStates > 0
        self.state_feature_name = self.State_Feature_Name if is_recurrent else None  
        features                = sorted( features + [ self.State_Feature_Name ] if is_recurrent else features ) 
        
                
        default_state = Config()
        default_state.depth    = 1
        default_state.width    = self.nStates
        default_idelta = Config()
        default_idelta.depth   = 1
        default_idelta.width   = self.nInst

        nOutput                 = self.nInst+self.nStates+self.nUpdateUnits
        self._layer             = DenseLayer( features=features, nOutput=nOutput, config=config.network, name=name+"_layer", dtype=dtype )
        self._init_state        = DenseLayer( features=sorted(state_features), nOutput=self.nStates, config=config.state.network, defaults=default_state, name=name+"_init_state", dtype=dtype ) if is_recurrent else None
        self._init_delta        = DenseLayer( features=sorted(init_delta_features), nOutput=self.nInst, config=config.init_delta.network, defaults=default_idelta, name=name+"_init_delta", dtype=dtype ) if init_delta else None
        config.done() # all config read
       
    def initial_state(self, features_time_0 : dict, training : bool = False ) -> tf.Tensor:
        """ Retrieves the initial state of the agent if the agent is recurrent """
        return self._init_state(features_time_0,training=training) if self.is_recurrent else None
    
    def initial_delta(self, features_time_0 : dict, training : bool = False ) -> tf.Tensor:
        """ Retrieves initial delta of the agent if requested """
        return self._init_delta(features_time_0,training=training) if not self._init_delta is None else None
    
    def call( self, all_features : dict, training : bool = False ) -> tuple:
        """
        Compute next action, and recurrent state.
        Function returns:
            ( tf.Tensor, tf.Tensor )
            
        Where:
            tf.Tensor: is the next action
            tf.Tensor: is the next state; or None
        """
        # if the mode is not recurrent -> just return next action
        if self.nStates == 0:
            return self._layer(all_features, training=training), None
        
        # recurrent mode
        # --------------
        
        def unit(x):
            if x is None:
                return None
            x = tf.math.sigmoid( x ) if self.sigmoid_1 else x
            return tf.where( x > 0.5, 1. , 0. )
        
        # impose limits on existing states 
        all_features    = dict(all_features)
        state           = all_features[self.State_Feature_Name]
        _log.verify( state.shape[1] == self.nStates, "Internal state '%s' should have second dimension %ld; found %ld", self.State_Feature_Name, self.nStates, state.shape[1] )
        
        def split_state(state, with_updcand ):
            state_sizes     = (self.classic_states, self.aggregate_states, self.past_repr_states, self.event_states )
            update_sizes    = (0,                   self.aggregate_states, self.past_repr_states, self.event_states )
            start_state     = 0
            start_update    = sum(state_sizes)
            out   = []
            for state_off, update_off in zip( state_sizes, update_sizes):
                state_here     = state[:,start_state:start_state+state_off] if state_off > 0 else None
                start_state    += state_off
                if not with_updcand:
                    out.append( state_here )
                else:
                    update_here    = state[:,start_update:start_update+update_off] if update_off > 0 else None
                    start_update   += update_off
                    out.append( (state_here, update_here ) )
            if with_updcand: 
                assert start_update == state.shape[1], "Internal error (1): only %ld of %ld states read." % (start_update, state.shape[1])
            else:
                assert start_state  == state.shape[1], "Internal error (2): only %ld of %ld states read." % (start_state, state.shape[1])
            return out
            
        split_states    = split_state(state, False)
        classic_state   = split_states[0]
        aggregate_state = split_states[1]
        past_repr_state = split_states[2]
        event_state     = split_states[3]

        # classic is simple
        classic_state   = tf.math.tanh( classic_state ) if not classic_state is None else None
        aggregate_state = tf.math.tanh( aggregate_state) if not aggregate_state is None and self.bound_aggr_states else aggregate_state
        event_state     = unit( event_state ) if not event_state is None else None
        
        # recompose
        state           = []
        if not classic_state is None:   state.append( classic_state )
        if not aggregate_state is None: state.append( aggregate_state )
        if not past_repr_state is None: state.append( past_repr_state )
        if not event_state is None:     state.append( event_state )
        state           = tf.concat( state, axis=1 ) if len(state) > 1 else state[0]
        assert state.shape[1] == self.nStates, "Internal error (3): should have %ld states not %ld" % (self.nStates, state.shape[1])

        all_features[self.State_Feature_Name] = state

        # execute
        output        = self._layer(all_features, training=training)
        out_action    = output[:,:self.nInst]
        out_recurrent = output[:,self.nInst:]
        assert out_recurrent.shape[1] == self.nStates + self.nUpdateUnits, "Internal error (4): expected length %ld but found %ld" % ( self.nStates + self.nUpdateUnits, out_recurrent.shape[1] )

        # process recurrence
        split_recurrent = split_state(out_recurrent, True)
        classic         = split_recurrent[0]
        aggregate       = split_recurrent[1]
        past_repr       = split_recurrent[2]
        event           = split_recurrent[3]

        # classic
        classic_state   = tf.math.tanh( classic[0] ) if not classic[0] is None else None
        
        # aggregate
        if not aggregate_state is None:
            candidate       = tf.math.tanh( aggregate[0] ) if self.bound_aggr_states else aggregate[0]
            update          = tf.math.sigmoid( aggregate[1] )
            aggregate_state = (1. - update) * aggregate_state + update * candidate

        # past_repr
        if not past_repr_state is None:
            candidate       = past_repr[0]
            update          = unit( past_repr[1] )
            past_repr_state = (1. - update) * past_repr_state + update * candidate

        # events
        if not event_state is None:
            candidate       = unit( event[0] )
            update          = unit( event[1] )
            event_state     = (1. - update) * event_state + update * candidate

        state           = []
        if not classic_state is None:   state.append( classic_state )
        if not aggregate_state is None: state.append( aggregate_state )
        if not past_repr_state is None: state.append( past_repr_state )
        if not event_state is None:     state.append( event_state )
        out_state           = tf.concat( state, axis=1 ) if len(state) > 1 else state[0]
        assert out_state.shape[1] == self.nStates, "Internal error (5): expected length %ld but found %ld" % ( self.nStates, out_state.shape[1] )

        return out_action, out_state

    @property
    def is_recurrent(self):
        """ Determines whether the current agent is recurrent, and has a 'state' """
        return not self._init_state is None
    @property
    def has_initial_delta(self):
        """ Whether the agent provides an initial delta (which still needs to be traded) """
        return not self._init_delta is None
    @property
    def nFeatures(self):
        """ Number of features used by the agent """
        return self._layer.nFeatures     
    @property
    def features(self):
        """ List of all features used by this agent. This includes the recurrent state, if the model is recurrent """
        return self._layer.features
    @property
    def public_features(self):
        """ Sorted list of all publicly visible features used by this agent. This excludes the internal recurrent state """
        return [ k for k in self._layer.features if not k == self.State_Feature_Name ]
    @property
    def available_features(self):
        """ List of features available to the agent """
        return [ k for k in self._layer.available_features if not k == self.State_Feature_Name ]
    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        assert not self._layer is None, "build() must be called first"
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] )

    @property
    def description(self):
        """ Returns a text description of 'self' """
        _log.verify( not self._layer is None, "build() must be called first")
        text_1 =         "Agent is using %ld weights: %ld for the main agent per step" % (self.num_trainable_weights, self._layer.num_trainable_weights)
        text_2 =         " Features available per time step:     %s\n" \
                         " Features used per time step:          %s" % ( fmt_list( self._layer.available_features ), fmt_list( self._layer.features ) )
        if self.has_initial_delta:
            text_1 +=   ", %ld for initial delta" % self._init_delta.num_trainable_weights
            text_2 +=  "\n Features available for initial delta: %s"\
                       "\n Features used by initial delta:       %s" % ( fmt_list( self._init_delta.available_features ), fmt_list( self._init_delta.features ) )
        if self.is_recurrent:
            text_1 +=   ", %ld for initial states" % self._init_state.num_trainable_weights
            text_2 +=  "\n Features available for initial state: %s"\
                       "\n Features used by initial state:       %s" % ( fmt_list( self._init_state.available_features ), fmt_list( self._init_state.features ) )
                
        return text_1 + ".\n" + text_2
    
# =========================================================================================
# Factory
# =========================================================================================

def AgentFactory( nInst : int, config : Config, name : str = None, dtype=dh_dtype ) -> tf.keras.layers.Layer:
    """
    Creates an agent network for nInst instruments based on 'config'.

    Parameters
    ----------
        nInst : int
            Number of instruments for deep hedging per time step
            
        config : Config
            Configuration. The most important is
                agent_type : str
                    Defines which agent to call. 
                    All other parameters of the config will
                    dependent on the agent chosen.                
        name : str, optional
            Namer of the tf layer for the agent
        per_step : bool, optional
            Whether the agent is used per time step, or once per sample.
            This allows the use of agents in other contexts, for example in the objective
            definition for y in OCE monetary utilities. See objectives.py
        dtype : tf.DType
            dtype

    Returns
    -------
        An agent model
    """    
    agent_type  = config("agent_type", "feed_forward", ['feed_forward', 'dense_agent'], "Which network agent type to use")
    agent       = None
    if agent_type in ["feed_forward", "dense_agent"]:
        agent = SimpleDenseAgent( nInst, config, name=name )
    
    _log.verify( not agent is None, "Unknnown agent type '%s'", agent_type )
    return agent
