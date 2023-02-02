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
        
        The agent's __call__ function will take in a dictionar of tenors
        of all feature avaialble, and return the corresponding action
        for 'nInst' instruments.
        
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
        self.nContStates        = config("recurrence",   0, Int>=0, "Number of real recurrent states. Set to zero to turn off recurrence")
        self.nDigitalStates     = config("recurrence01", 0, Int>=0, "Number of digital recurrent states. Set to zero to turn off recurrence")
        self.like_gru           = config("recurrence_gru", False, bool, "If True, adds a GRU-like update state mechanism")
        self.nInst              = int(nInst)
        self.nStates            = self.nContStates + self.nDigitalStates
        
        default_state = Config()
        default_state.depth    = 1
        default_state.width    = self.nStates
        default_idelta = Config()
        default_idelta.depth   = 1
        default_idelta.width   = self.nInst
 
        _log.verify( self.State_Feature_Name not in features, "Cannot use internal state name '%s' in feature list", self.State_Feature_Name )
        
        is_recurrent            = self.nStates > 0
        self.state_feature_name = self.State_Feature_Name if is_recurrent else None  
        features                = sorted( features + [ self.State_Feature_Name ] if is_recurrent else features ) 
        
        nOutput                 = self.nInst
        if is_recurrent:
            if not self.like_gru:
                nOutput         = self.nInst+self.nStates
            else:
                nOutput         = self.nInst+2*self.nStates
                
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
        
        # map states into [-1,+1]
        all_features = dict(all_features)
        state        = all_features[self.State_Feature_Name]
        state        = tf.math.tanh(state, name="tanh_state")
        
        # handle digital states
        if self.nDigitalStates > 0:
            cont_state = state[:,:self.nContStates] if self.nContStates > 0 else None
            digi_state = state[:,self.nContStates:]
            digi_state = tf.where( digi_state >= 0., 1., 0., name="digital_state" )
            state      = tf.concat( [cont_state,digi_state], axis = 1, name="cont_digital_state") if self.nContStates > 0 else digi_state
        all_features[self.State_Feature_Name] = state

        # execute
        output      = self._layer(all_features, training=training)
        out_action  = output[:,:self.nInst]
        out_state   = output[:,self.nInst:]
        
        # GRU mode
        # We are not quite applying GRU, but something similar
        if self.like_gru:
            assert out_state.shape[1] == self.nStates*2, "Internal error: expected %ld output size, found %ld" % ( self.nStates*2, out_state.shape[1] )
            control   = out_state[:,self.nStates:]
            out_state = out_state[:,:self.nStates]
            control   = tf.math.sigmoid(control)
            out_state = state * control + (1.-control) * out_state
            
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
            text_1 +=   ", %ld for the initial state" % self._init_state.num_trainable_weights
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
