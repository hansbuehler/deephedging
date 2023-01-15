# -*- coding: utf-8 -*-
"""
Deep Hedging Agents
-------------------
Contains simple training agents

June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype, Int
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
        features                = config("features", self.Default_features_per_step, list, help="Named features for the agent to use")
        self.nStates            = config("recurrence", 0, Int>=0, "Number of recurrent states. Set to zero to turn off recurrence")
        if self.nStates > 0:
            features.append( self.State_Feature_Name )
            features = sorted(features)
        self.state_feature_name = self.State_Feature_Name if self.nStates > 0 else None
        self.nInst              = int(nInst)
        self.layer              = DenseLayer( features=features, nOutput=self.nInst+self.nStates, config=config.network, name=name+"_dense", dtype=dtype )
        self.init_state         = VariableLayer( (self.nStates,), name=name+"_initial_state" if not name is None else None, dtype=dtype ) if self.nStates > 0 else None
        config.done() # all config read
        
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
            return self.layer(all_features, training=training), None
        
        # if the model is recurrent, handle recurrent state
        if not self.State_Feature_Name in all_features:
            # TF expects the leading dimension for any variable to be the number of samples.
            # Below is a bit of a hack
            all_features    = dict(all_features)
            delta           = all_features['delta']   # current spot [None,N]
            init_state      = self.init_state(all_features, training=training)
            assert len(delta.shape) == 2, "'delta': expected tensor of dimensionm 2. Found %s" % delta.shape.as_list()
            assert len(init_state.shape) == 1, "init_state: expected to be a plain vector. Found shape %s" % init_state.shape
            init_state      = delta[:,0][:,tf.newaxis] * 0. + init_state[tf.newaxis,:] 
            all_features[self.State_Feature_Name] = init_state
        
        action_state = self.layer(all_features, training=training)
        return action_state[:,:self.nInst], action_state[:,self.nInst:]

    @property
    def is_recurrent(self):
        """ Determines whether the current agent is recurrent, and has a 'state' """
        return not self.init_state is None
    @property
    def nFeatures(self): # NOQA
        """ Number of features used by the agent """
        return self.layer.nFeatures     
    @property
    def features(self):
        """ List of all features used by this agent. This includes the recurrent state, if the model is recurrent """
        return self.layer.features
    @property
    def public_features(self):
        """ Sorted list of all publicly visible features used by this agent. This excludes the internal recurrent state """
        return [ k for k in self.layer.features if not k == self.State_Feature_Name ]
    @property
    def available_features(self): # NOQA
        """ List of features available to the agent """
        return [ k for k in self.layer.available_features if not k == self.State_Feature_Name ]
    
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
