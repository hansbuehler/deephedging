# -*- coding: utf-8 -*-
"""
Softclip
--------
Objectives, mostly monetary utilities see forthcoming book learning-to-trade.com
June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype, Int, Float, tfCast
from .agents import AgentFactory
from .objectives import MonetaryUtility
import tensorflow_probability as tfp # NOQA
from collections.abc import Mapping
from cdxbasics.util import uniqueHash

_log = Logger(__file__)

class DHSoftClip(tf.keras.layers.Layer):
    """
    Simple wrapper around tensorflow_probability.bijectors.SoftClip which also provides for hard clips
    TODO: remove dependency on tensorflow_probability
    """

    def __init__(self, config, name : str = None, dtype : tf.DType = dh_dtype ):
        """
        Initialize softclip from tensorflow_probability
        """
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )        
    
        self.hard_clip             = config('hard_clip', False, bool, "Use min/max instread of soft clip for limiting actions by their bounds")
        self.outer_clip            = config('outer_clip', True, bool, "Apply a hard clip 'outer_clip_cut_off' times the boundaries")
        self.outer_clip_cut_off    = config('outer_clip_cut_off', 10., Float>=1., "Multiplier on bounds for outer_clip")
        hinge_softness             = config('softclip_hinge_softness', 1., Float>0., "Specifies softness of bounding actions between lbnd_a and ubnd_a")
        self.softclip              = tfp.bijectors.SoftClip( low=0., high=1., hinge_softness=hinge_softness, name='soft_clip' if name is None else name )
        config.done()
    
    def __call__( self, actions, lbnd_a, ubnd_a ):
        """ Clip the action within lbnd_a, ubnd_a """
        
        with tf.control_dependencies( [ tf.debugging.assert_greater_equal( ubnd_a, lbnd_a, message="Upper bound for actions must be bigger than lower bound" ),
                                        tf.debugging.assert_greater_equal( ubnd_a, 0., message="Upper bound for actions must not be negative" ),
                                        tf.debugging.assert_less_equal( lbnd_a, 0., message="Lower bound for actions must not be positive" ) ] ):
        
            if self.hard_clip:
                # hard clip
                # this is recommended for debugging only.
                # soft clipping should lead to smoother gradients
                actions = tf.minimum( actions, ubnd_a, name="hard_clip_min" )
                actions = tf.maximum( actions, lbnd_a, name="hard_clip_max" )
                return actions            

            if self.outer_clip:
                # to avoid very numerical errors due to very
                # large pre-clip actions, we cap pre-clip values
                # hard at 10 times the bounds.
                # This can happen if an action has no effect
                # on the gains process (e.g. hedge == 0)
                actions = tf.minimum( actions, ubnd_a*self.outer_clip_cut_off, name="outer_clip_min" )
                actions = tf.maximum( actions, lbnd_a*self.outer_clip_cut_off, name="outer_clip_max" )

            dbnd = ubnd_a - lbnd_a
            rel  = ( actions - lbnd_a ) / dbnd
            rel  = self.softclip( rel )
            act  = tf.where( dbnd > 0., rel *  dbnd + lbnd_a, 0., name="soft_clipped_act" )
            act  = tf.debugging.check_numerics(act, "Numerical error clipping action in %s. Turn on tf.enable_check_numerics to find the root cause. See trainer.py" % __file__ )
            return act
