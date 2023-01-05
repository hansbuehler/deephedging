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
    The basic functionality is a generic pattern, e.g.
    
        - The __init__ function takes a config object which contains all relevant data for the action
        - build() and call() expect dictionaries
        - The 'features' attribute has a sorted list of required model features, by name.
          The action will attempt to extract those features whenever it is called 
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
        features           = config("features", self.Default_features_per_step, list, help="Named features for the agent to use")
        self.nStates       = config("recurrence", 0, Int>=0, "Number of recurrent states. Set to zero to turn off recurrence")
        if self.nStates > 0:
            features.append( self.State_Feature_Name )
        self.nInst         = int(nInst)
        self.layer         = DenseLayer( features=features, nOutput=self.nInst+self.nStates, config=config.network, name=name+"_dense", dtype=dtype )
        self.init_state    = VariableLayer( (self.nStates,), name=name+"_initial_state" if not name is None else None, dtype=dtype ) if self.nStates > 0 else None
        config.done() # all config read
        
    @property
    def nFeatures(self): # NOQA
        return self.layer.nFeatures 
        
    def call( self, all_features : dict, training : bool = False ) -> tuple:
        """
        Compute next action, and recurrent state.
        Function returns:
            ( tf.Tensor, dict )
            
        Where:
            tf.Tensor: is the next action
            dict: is a dictionary which is to be merged with the feature space for the next call of this layer
        """
        # if the mode is not recurrent -> just return next action
        if self.nStates == 0:
            return self.layer(all_features, training=training), {}
        
        # if the model is recurrent, handle recurrent state
        if not self.State_Feature_Name in all_features:
            all_features = dict( all_features )
            all_features[self.State_Feature_Name] = self.init_state(all_features, training=training)
        
        action_state = self.layer(all_features, training=training)
        return action_state[:,:self.nInst], { self.State_Feature_Name : action_state[:,self.nInst:] }


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
