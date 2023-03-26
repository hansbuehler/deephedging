# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for

June 30, 2022
@author: hansbuehler
"""

from cdxbasics.prettydict import PrettyDict
from cdxbasics.util import uniqueHash, fmt_list, fmt_big_number, fmt_seconds, fmt_now
from cdxbasics.np import mean, err
from cdxbasics.config import Config, Int, Float
from cdxbasics.verbose import Context
from cdxbasics.subdir import SubDir, uniqueFileName48, CacheMode
from cdxbasics.logger import Logger
import time as time
import numpy as np # NOQA
import psutil as psutil
import inspect as inspect
import os as os

from packages.cdx_tf.cdx_tf.util import tf, npCast, tfCast, TF_VERSION, def_dtype
from packages.cdx_tf.cdx_tf.gym import ProgressData, Environment, TrainingInfo, train as cdx_train, Status
from packages.cdx_tf.cdx_tf.layers import DenseAgent, RecurrentAgent
from packages.cdx_tf.cdx_tf.clip import SoftClip
from packages.cdx_tf.cdx_tf.monetary_utility import MonetaryUtility
from .gym2 import VanillaDeepHedgingGym

_log = Logger(__file__)

# =========================================================================================
# TrainingProgressData
# Implements the data sets tracked by the training program.
# =========================================================================================

class TrainingProgressData(ProgressData):
    """
    Class to keep track of data for printing progress during training
    This object is serialized to upon caching or when multi-process training makes a GUI update.

    We add tracking of the utility values
    """

    def __init__(self, environment     : Environment,        # gym, tf_data, etc
                       predicted_data0 : PrettyDict,         # results
                       training_info   : TrainingInfo,       # total number of epochs requested etc
                       config          : Config              # configuration, if any
                       ):
        ProgressData.__init__(  self,
                                environment         = environment,
                                predicted_data0     = predicted_data0,
                                training_info       = training_info,
                                config              = config,
                                best_by_training    = True,
                                store_epoch_results = True,
                                store_best_results  = True)

        # keep track of utility values
        self.trn.utility_losses     = []
        self.trn.utility_loss_errs  = []
        self.trn.utility0_loss      = []
        self.trn.utility0_loss_err  = []
        if not predicted_data0.val is None:
            self.val.utility_loss      = []
            self.val.utility_loss_err  = []
            self.val.utility0_loss     = []
            self.val.utility0_loss_err = []

        self.on_epoch_end_prep(  environment    = environment,
                                 predicted_data = predicted_data0,
                                 training_info  = training_info,
                                 logs           = {}
                        )

    def on_epoch_end_prep(self,  environment    : Environment,  # gym, tf_data, etc
                                 predicted_data : PrettyDict,   # current predicted training and validation data; current loss.
                                 training_info  : TrainingInfo, # number of epochs to be computed etc
                                 logs           : dict          # logs c.f. keras Callback
                        ) -> int:

        trnP = environment.trn.sample_weights
        valP = environment.val.sample_weights if not environment.val is None else None

        self.trn.utility_losses.append( mean( trnP, predicted_data.trn.results.utility ) )
        self.trn.utility_loss_errs.append( err(  trnP, predicted_data.trn.results.utility ) )
        self.trn.utility0_loss.append( mean( trnP, predicted_data.trn.results.utility0) )
        self.trn.utility0_loss_err.append( err(  trnP, predicted_data.trn.results.utility0 ) )
        if not predicted_data.val is None:
            self.val.utility_loss.append( mean( valP, predicted_data.val.results.utility ) )
            self.val.utility_loss_err.append( err(  valP, predicted_data.val.results.utility ) )
            self.val.utility0_loss.append( mean( valP, predicted_data.val.results.utility0 ) )
            self.val.utility0_loss_err.append( err(  valP, predicted_data.val.results.utility0 ) )

        return Status.CONTINUE

    """
    def on_epoch_end_update( self, environment    : Environment,    # gym, tf_data, etc
                                   training_info  : TrainingInfo ): # number of epochs to be computed etc)
        status_message = self.status_message( environment=environment, training_info=training_info )
        environment.verbose.report(0, status_message)

    def on_done( self,      environment    : Environment,  # gym, tf_data, etc
                            predicted_data : PrettyDict,   # current predicted training and validation data; current loss.
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                            stop_reason    : Status    # why training stopped
                        ):
        status_message = self.done_message( environment=environment, training_info=training_info, stop_reason=stop_reason )
        environment.verbose.report(0, status_message)
    """

# =========================================================================================
# Main training loop
# =========================================================================================

def train2( world,
            val_world,
            config  : Config = Config(),
            verbose : Context = Context(verbose="all") ):
    """
    Train our deep hedging model with with the provided world.
    Main training loop.

    Parameters
    ----------
        world     : world with training data
        val_world : world with validation data (e.g. computed using world.clone())
        config    : configuration, including config.gym for the gym

    Returns
    -------
        A PrettyDict which contains, computed at the best weights:
            gym           : trained gym, set to best weights.
            progress_data : progress data of type TrainingProgressData.
                            This also contains timing information.
        The dictionary also contains two sub-pretty dictrionaryes 'trn' and optionally 'val':
            trn.result   : numpy arrays of the training results from gym(trn.tf_data) in the best point
            trn.loss     : float of the training loss for the current gym, e.g. np.mean( trn.result.loss ) if sample_weights are not used, otherwise the corresponding calculation.
            trn.loss_err : float of the standard error of the training loss (e.g. std/sqrt(n))
        If val is not None:
            val.result   : numpy arrays of the validation results from gym(val.tf_data) in the best point
            val.loss     : float of the validation loss for the current gym
            val.loss_err : float of the standard error of the validation loss (e.g. std/sqrt(n))

    The current training loop set up is a bit messy between managing user feedback, and also allowing to cache results
    to support warm starting. Will at some point redesign this architecture to create cleaner delineation of data, caching,
    and visualization (at the very least to support multi-processing training)
    """

    env = Environment(
               gym                   = VanillaDeepHedgingGym(config.gym),
               tf_trn_data           = world.tf_data,
               tf_val_data           = val_world.tf_data,
               tf_sample_weights     = world.tf_sample_weights,
               tf_val_sample_weights = val_world.tf_sample_weights
               )
    env.trn.world = world
    env.val.world = val_world

    r = cdx_train( environment           = env,
                   create_progress       = TrainingProgressData,
                   config                = config.trainer,
                   verbose               = verbose
             )

    return r


