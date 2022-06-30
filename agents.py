# -*- coding: utf-8 -*-
"""
Deep Hedging Agents
-------------------
Contains simple training agents

June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype
from collections.abc import Mapping
_log = Logger(__file__)

class FeedForwardAgent(tf.keras.layers.Layer):
    """
    Simple Action Model for Deep Hedging
    The basic functionality is a generic pattern, e.g.
    
        - The __init__ function takes a config object which contains all relevant data for the action
        - build() and call() expect dictionaries
        - The 'features' attribute has a sorted list of required model features, by name.
          The action will attempt to extract those features whenever it is called 
    """
    
    Default_features = [ 'price', 'delta', 'time_left' ]

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
            dtype : tf.DType, optional
                dtype
        """
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )
        self.nInst         = int(nInst)
        self.width         = config.network("width",20, help="Network width.")
        self.activation    = config.network("activation","relu", help="Network activation function")
        self.depth         = config.network("depth", 3, int, help="Network depth")
        self.features      = sorted( set( config("features", self.Default_features, list, help="Named features the agent uses from the environment") ) )
        self.model         = None
        self.nFeatures     = None
        config.done() # all config read
        
    def build( self, shapes : dict ):
        """ 
        Keras layer builld() function.
        'shapes' must be a dictionary
        """
        _log.verify( isinstance(shapes, Mapping), "'shapes' must be a dictionary type. Found type %s", type(shapes ))
        assert self.nFeatures is None and self.model is None, ("build() call twice")
        # collect features
        self.nFeatures = 0
        for feature in self.features:
            _log.verify( feature in shapes, "Unknown feature '%s'. Known features are: %s", feature, list(shapes) )            
            fs = shapes[feature]
            assert len(fs) == 2, ("Internal error: all features should have been flattend. Found feature '%s' with shape %s" % (feature, fs))
            self.nFeatures += fs[1]
        # build model
        # simple feedforward model as an example
        if self.nFeatures == 0:
            _log.warn("No features for the Auction selected")
            return
        inp = tf.keras.layers.Input( shape=(self.nFeatures,), dtype=self.dtype )
        x = inp
        for d in range(self.depth):
            x = tf.keras.layers.Dense( units=self.width,
                                       activation=self.activation,
                                       use_bias=True )(x)
        x = tf.keras.layers.Dense(     units=self.nInst,
                                       activation="linear",
                                       use_bias=True )(x)
        self.model         = tf.keras.Model( inputs=inp, outputs=x )
        
    def call( self, data : dict, training : bool = False ) -> tf.Tensor:
        """
        Ask the agent for an action.
    
        Parameters
        ----------
            data : dict
                Contains all available features at this time step.
                This must be a dictionary.
            training : bool, optional
                Whether we are training or not
                
        Returns
        -------
            Tensor with actions. The second dimension of
            the tensor corresponds to self.nInst
    
        """
        _log.verify( isinstance(data, Mapping), "'data' must be a dictionary type. Found type %s", type(data ))
        assert not self.nFeatures is None, ("Model not build()")
        if self.model is None:
            return 0.  
        # run model
        features = [ data[_] for _ in self.features ]
        features = tf.concat( features, axis=1, name = "features" )      
        assert self.nFeatures == features.shape[1], ("Condig error: number of features should match up. Found %ld and %ld" % ( self.nFeatures, features.shape[1] ) )
        return self.model( features, training=training )
       
    
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
        dtype : tf.DType
            dtype

    Returns
    -------
        An agent model
    """    
    agent_type  = config("agent_type", "feed_forward", str)
    agent       = None
    if agent_type == "feed_forward":
        agent = FeedForwardAgent( nInst, config, name=name, dtype=dtype )
    
    _log.verify( not agent is None, "Unknnown agent type '%s'", agent_type )
    return agent
