# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for 

June 30, 2022
@author: hansbuehler
"""

#from .base import Logger, npCast, fmt_seconds, mean, err, tf, mean_bins, mean_cum_bins, perct_exp, Int, Float, fmt_big_number, fmt_list
from .base import Logger, Config, tf, Int, Float, mean, err, npCast, fmt_list, fmt_big_number, fmt_seconds, TF_VERSION#NOQA
from .plot_training import Plotter
from cdxbasics.prettydict import PrettyDict as pdct
from cdxbasics.util import uniqueHash
from cdxbasics.config import Config
from cdxbasics.subdir import SubDir, uniqueFileName48, CacheMode
import time as time
import numpy as np # NOQA
import psutil as psutil
import inspect as inspect
import os as os

_log = Logger(__file__)

# =========================================================================================
# Monitor
# =========================================================================================

class TrainingInfo(object):
    """ Information on the current training run """
    
    def __init__(self, *, batch_size, epochs, output_level, num_weights):#NOQA
        self.epochs       = epochs       # epochs requested by the user. Note that the actual epochs will be reduced by the number of epochs already trained in the cached file
        self.batch_size   = batch_size   # batch size.
        self.output_level = output_level # one of: 'quiet', 'text', 'all'
        self.num_weights  = num_weights  # number of trainable weights
        assert self.output_level in ['quiet', 'text', 'all'], "Invalid 'output_level': should be 'quiet', 'text', or 'all'. Found %s" % output_level

class TrainingProgressData(object):
    """
    Class to keep track of data for printing progress during training
    This object is serialized to desk upon caching
    """
    
    def __init__(self, *, gym, world, val_world, result0 ):
        """ Initialize data """
        self.result0          = result0                     # initil results from calling gym() on the training set
        self.training_result  = None                        # full results corresponding to current weights, training set
        self.val_result       = None                        # full results corresponding to current weights, validation set
        self.times            = []                          # times per epoch
        
        # track progress
        self.losses            = pdct()
        self.losses.batch      = []                         # losses for last batch (from tensorflow)
        self.losses.training   = []                         # losses for training set (manually computed)
        self.losses.val        = []                         # losses for validation set (manually computed)
        self.losses_err        = pdct()
        self.losses_err.training = []                       # std error for training loss
        self.losses_err.val      = []                       # std error for validation loss

        self.init_loss         = mean( world.sample_weights, result0.loss )
        self.init_loss_err     = err( world.sample_weights, result0.loss )
        self.best_loss         = self.init_loss 
        self.best_loss_err     = self.init_loss_err
        self.best_weights      = gym.get_weights()
        self.best_epoch        = -1
        
        self.utilities         = pdct()
        self.utilities.training_util      = []           # utility value of the hedged payoff for training set
        self.utilities.training_util0     = []           # utility value of the payoff for training set
        self.utilities.training_util_err  = []           # errors for the above
        self.utilities.training_util0_err = []
        self.utilities.val_util           = []           # validation set for the above
        self.utilities.val_util0          = []
        
        # process information: this vector is len+1 as it includes the initial state
        self.process = pdct()
        p = psutil.Process()
        with p.oneshot():
            self.process.memory_rss = [ p.memory_info().rss / (1024.*1024.) ]
            self.process.memory_vms = [ p.memory_info().vms / (1024.*1024.) ]
                
    @property
    def epoch(self):
        """ Returns the current epoch. Returns -1 if no epoch was yet recorded """
        return len(self.times)-1

    def on_epoch_end( self, *, gym, world, val_world, loop_epoch, time_epoch, batch_loss ):
        """ Update data set with the latest results """
        self.training_result = npCast( gym(world.tf_data) )
        self.val_result      = npCast( gym(val_world.tf_data) )
        self.times.append( time_epoch )

        # losses
        # Note that we apply world.sample_weights to all calculations
        # so we are in sync with keras.fit()
        self.losses.batch.append(       batch_loss )
        self.losses.training.append(    mean(world.sample_weights, self.training_result.loss) )
        self.losses.val.append(         mean(val_world.sample_weights, self.val_result.loss) )
        self.losses_err.training.append( err(world.sample_weights,self.training_result.loss) )
        self.losses_err.val.append(      err(val_world.sample_weights,self.val_result.loss) )
        
        # utilities
        self.utilities.training_util.append(     mean(world.sample_weights, self.training_result.utility ) )
        self.utilities.training_util0.append(    mean(world.sample_weights, self.training_result.utility0) )
        self.utilities.training_util_err.append( err( world.sample_weights, self.training_result.utility ) )
        self.utilities.training_util0_err.append(err( world.sample_weights, self.training_result.utility0) )
        self.utilities.val_util.append(      mean(val_world.sample_weights, self.val_result.utility ) )
        self.utilities.val_util0.append(     mean(val_world.sample_weights, self.val_result.utility0 ) )
        
        # store best loss
        if self.losses.training[-1] < self.best_loss:
            self.best_loss         = self.losses.training[-1]
            self.best_loss_err     = self.best_loss_err
            self.best_weights      = gym.get_weights()
            self.best_epoch        = self.epoch
            
        # memory usage
        p = psutil.Process()
        with p.oneshot():
            self.process.memory_rss.append( p.memory_info().rss / (1024.*1024.))
            self.process.memory_vms.append( p.memory_info().vms / (1024.*1024.) )
            
    def set_best_weights(self, *, gym, world, val_world):
        """
        Write best weights into gym and set current state accordingly
        The function updates 'training_result' and 'val_result', too
        """        
        gym.set_weights( self.best_weights )
        self.training_result = npCast( gym(world.tf_data) )
        self.val_result      = npCast( gym(val_world.tf_data) )


class Monitor(tf.keras.callbacks.Callback):
    """
    Manages training of our model    
    -- Keeps track of training data in TrainingProgressData including best fit
    -- Implements caching
    -- Implements dyanmic visual updates
    """
    
    def __init__(self, *, gym, world, val_world, result0, training_info, config = Config(), output_level = "all" ):# NOQA
        tf.keras.callbacks.Callback.__init__(self)
        
        self.gym              = gym  
        self.world            = world
        self.val_world        = val_world
        self.training_info    = training_info
        self.why_stopped      = "Ran all epochs"
        self.epoch_start      = None
        self.time0            = None
        self.cache_last_epoch = -1
        self.is_aborted       = False

        self.cache_dir        = config.caching("directory", "./.deephedging_cache", str, "If specified, will use the directory to store a persistence file for the model")
        self.cache_mode       = config.caching("mode", CacheMode.ON, CacheMode.MODES, "Caching strategy: %s" % CacheMode.HELP)
        self.cache_freq       = config.caching("epoch_freq", 10, Int>0, "How often to cache results, in number of epochs")
        cache_file_name       = config.caching("debug_file_name", None, help="Allows overwriting the filename for debugging an explicit cached state")
        self.plotter          = Plotter(training_info.output_level == 'all', config.visual) if training_info.output_level != 'quiet' else None
        config.done()
                
        self.progress_data    = TrainingProgressData(    
                                        gym            = gym, 
                                        world          = world, 
                                        val_world      = val_world,
                                        result0        = result0
                                        )
        
        if not self.plotter is None: print("Network feature information:\n"\
                " Features used by the agent:        %s\n"\
                " Features available to the agent:   %s\n"\
                " Features used by the utility:      %s\n"\
                " Features available to the utility: %s" % \
            ( fmt_list(gym.agent_features_used), fmt_list(gym.available_features_per_step), fmt_list(gym.utility_features_used), fmt_list(gym.available_features_per_path)) )

        # caching
        self.cache_mode       = CacheMode( self.cache_mode )
        self.cache_dir        = SubDir(self.cache_dir, "!")
        optimizer_id          = uniqueHash( tf.keras.optimizers.serialize( gym.optimizer ) )
        self.cache_file       = uniqueFileName48( gym.unique_id, optimizer_id, world.unique_id, val_world.unique_id ) if cache_file_name is None else cache_file_name
        self.full_cache_file  = self.cache_dir.fullKeyName( self.cache_file )

        if not self.cache_mode.is_off:
            if not self.plotter is None: print("Caching enabled @ '%s'" %  self.full_cache_file)
            if self.cache_mode.delete:
                self.cache_dir.delete( self.cache_file )    
            elif self.cache_mode.read:
                # restore cache                
                cache = self.cache_dir.read( self.cache_file )
                if not cache is None:
                    # load everything except the gym 
                    # restore gym
                    if not self.gym.restore_from_cache( cache['gym'] ):
                        if not self.plotter is None: print(\
                              "\rCache consistency error: could not write weights from cache to current model. This is most likely because the model architecture changed.\n"\
                              "Use config.train.caching.mode = 'renew' to rebuild the cache if this is the case. Use config.train.caching.mode = 'off' to turn caching off.\n")
                    else:
                        self.progress_data = cache['progress_data']
                        _log.verify( self.progress_data.epoch >= 0, "Error: object restored from cache had epoch set to %ld", self.progress_data.epoch )
                        self.cache_last_epoch = self.progress_data.epoch
                        if not self.plotter is None: print("Cache successfully loaded. Current epoch: %ld" % self.progress_data.epoch )

        # initialize timing
        if self.progress_data.epoch+1 >= training_info.epochs:
            if not self.plotter is None: print( \
                   "Nothing to do: cached model loaded from %s was trained for %ld epochs; you have asked to train for %ld epochs. "\
                   "Raise number of epochs or turn off caching to re-start training.\n\nPlotting results for the trained model.\n" % \
                   ( self.full_cache_file, self.progress_data.epoch+1, training_info.epochs ) )
        self.time0 = time.time()

    @property
    def is_done(self):
        """ Checks whether training has finished. This can happen at inception if a cache is restored which was trained for as many epochs as requested """
        return self.progress_data.epoch+1 >= self.training_info.epochs
    
    @property
    def current_epoch(self):
        """ Returns the current epoch. -1 if no epoch was run """
        return self.progress_data.epoch
    
    def on_epoch_begin( self, epoch, logs = None ):
        """ If this is the first epoch, tell user we started training. """
        if self.progress_data.epoch == -1:
            weights    = fmt_big_number( self.gym.num_trainable_weights )
            act_epochs = self.training_info.epochs-(self.progress_data.epoch+1)
            if not self.plotter is None: print("Deep Hedging Engine: first of %ld epochs for training %s weights over %ld samples with %ld validation samples started. This training run took %s so far. Now compiling graph ...       " % (act_epochs, weights, self.world.nSamples, self.val_world.nSamples, fmt_seconds(time.time()-self.time0)), end='')
        self.epoch_start      = time.time()
            
    def on_epoch_end( self, loop_epoch, logs = None ):
        """
        Called when an epoch ends
        Handle plotting, and caching
        Note that 'loop_epoch' is the epoch of the current training run. If the state was recovered from a cache, it won't be the logical epoch
        """
        if self.progress_data.epoch == -1:
            empty = " "*200
            if not self.plotter is None: print("\r\33[2K "+empty+"\r", end='')
        
        self.progress_data.on_epoch_end( 
                                gym        = self.gym, 
                                world      = self.world, 
                                val_world  = self.val_world,
                                loop_epoch = loop_epoch,
                                time_epoch = time.time() - self.epoch_start,
                                batch_loss = float( logs['loss_default_loss'] ), # we read the metric instead of 'loss' as this appears to be weighted properly
                                )
        assert self.progress_data.epoch >= 0, "Internal error"
        
        # cache or not
        # ------------
        
        if self.current_epoch % self.cache_freq == 0 and self.cache_mode.write and self.current_epoch > self.cache_last_epoch:
            self.write_cache()
        
        # plot
        # ----
        
        if self.plotter is None:
            return
        self.plotter(world             = self.world, 
                     val_world         = self.val_world, 
                     last_cached_epoch = self.cache_last_epoch,
                     progress_data     = self.progress_data, 
                     training_info     = self.training_info )

    def finalize( self, status ):
        """
        Close training. Call this even if training was aborted
        -- Cache the current state
        -- Apply best weight
        """
        # tell user what happened
        empty = " "*200
        if not self.plotter is None: print("\r\33[2K"+ ( "*** Aborted *** " if self.is_aborted else "") + empty, end='')

        # cache current state /before/ we reset gym to its best weights
        # this way we can continue to train from where we left it
        cached_msg = ""
        if self.progress_data.epoch >= 0 and self.cache_mode.write:
            self.write_cache()
            cached_msg = " State of training until epoch %ld cached into %s\n" % (self.cache_last_epoch, self.full_cache_file)

        # restore best weights
        self.progress_data.set_best_weights( gym=self.gym, world=self.world, val_world=self.val_world )

        # upgrade plot
        if not self.plotter is None:
            self.plotter(world             = self.world, 
                         val_world         = self.val_world, 
                         last_cached_epoch = self.cache_last_epoch,
                         progress_data     = self.progress_data, 
                         training_info     = self.training_info )
            self.plotter.close()
                
        if not self.plotter is None: print("\n Status: %s.\n Weights set to best epoch: %ld\n%s" % (status, self.progress_data.best_epoch,cached_msg) )
    
    def write_cache(self):
        """ Write cache to disk """
        cache = { 'gym':           self.gym.create_cache(),
                  'progress_data': self.progress_data
                }            
        self.cache_dir.write( self.cache_file, cache )
        self.cache_last_epoch = self.progress_data.epoch
        
# =========================================================================================
# training
# =========================================================================================

def create_optimizer( train_config : Config ):
    """
    Creates an optimizer from a config object
    The keywords accepted are those documented for https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    
    You can use:
        config.optimizer = "adam"
        config.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
        
    Or the new form for TF2.11 optimizers
    
    """    
    # legacy 1.0 support
    if 'optimizer' in train_config:
        return train_config("optimizer", "RMSprop", help="Optimizer" )

    # new version. Specify optimizer.name
    config    = train_config.optimizer
    name      = config("name", "adam", str, "Optimizer name. See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers")

    # auto-detect valid parameters
    optimizer = tf.keras.optimizers.get(name)
    sig_opt   = inspect.signature(optimizer.__init__)
    classname = optimizer.__class__
    kwargs    = {}
    
    # all parameters requested by the optimizer class
    for para in sig_opt.parameters:
        if para in ['self','name','kwargs']:
            continue
        default = sig_opt.parameters[para].default
        if default == inspect.Parameter.empty:
            # mandatory parameter
            kwargs[para] = config(para, help="Parameter %s for %s" % (para,classname))
        else:
            # optional parameter
            kwargs[para] = config(para, default, help="Parameter %s for %s" % (para,classname))

    # The following parameters are understood by general tensorflow optimziers
    hard_coded = dict(  clipnorm=None,
                        clipvalue=None,
                        global_clipnorm=None )
    if TF_VERSION >= 211:
        hard_coded.update(
                        use_ema=False,
                        ema_momentum=0.99,
                        ema_overwrite_frequency=None)
    for k in hard_coded:
        if k in kwargs:
            continue  # handled already
        v = hard_coded[k]
        kwargs[k] = config(k, v, help="Parameter %s for keras optimizers" % k)
    
    config.done()
    return optimizer.__class__(**kwargs)
    
    
def default_loss( y_true,y_pred ):     
    """ Default loss: ignore y_true """
    return y_pred

# =========================================================================================

def train(  gym,
            world,
            val_world,
            config  : Config = Config() ):
    """ 
    Train our deep hedging model with with the provided world.
    Main training loop.
    
    V2.0: this function now support caching.
    It should recover somewhat transparently in its default setup.
    WARNING: if you change model logic, make sure to flush the cache with config.caching.mode = "renew"
    
    Parameters
    ----------
        gym       : VanillaDeepHedgingGym or similar interface
        world     : world with training data
        val_world : world with validation data (e.g. computed using world.clone())
        config    : configuration

    Returns
    -------
        Nothing.        
            Run gym(world) to get trained data for the full data set.
            The gym itself contains the weights of the trained agents.
            
    The current training loop set up is a bit messy between managing user feedback, and also allowing to cache results
    to support warm starting. Will at some point redesign this architecture to create cleaner delineation of data, caching, 
    and visualization (at the very least to support multi-processing training)
    """
    tf.debugging.enable_check_numerics()
    
    # how much to print
    output_level     = config("output_level", "all", ['quiet', 'text', 'all'], "What to print during training")

    # training parameters    
    batch_size       = config.train("batch_size",  None, help="Batch size")
    epochs           = config.train("epochs",      100, Int>0, help="Epochs")
    run_eagerly      = config.train("run_eagerly", False, help="Keras model run_eagerly. Turn to True for debugging. This slows down training. Use None for default.")
    learning_rate    = config.train("learing_rate", None, help="Manually set the learning rate of the optimizer")
    optimzier        = create_optimizer(config.train)
    
    # tensorboard: have not been able to use it .. good luck.
    tboard_log_dir   = config.train.tensor_board(   "log_dir", "", str, "Specify tensor board log directory")
    tboard_freq      = config.train.tensor_board(   "hist_freq", 1, Int>0, "Specify tensor board log frequency") 
    tboard_prf_batch = config.train.tensor_board(   "profile_batch", 0, Int>0, "Batch used for profiling. Set to non-zero to activate profiling") 
    
    t0               = time.time()
    result0          = gym(world.tf_data)   # builds the model
    gym.compile(    optimizer        = optimzier, 
                    loss             = dict( loss=default_loss ),
                    weighted_metrics = dict( loss=default_loss ),
                    run_eagerly      = run_eagerly)
    if not learning_rate is None:
        gym.optimizer.lr = float( learning_rate )
    if output_level != "quiet": print("Gym with %s trainable weights compiled and initialized. Took %s" % (fmt_big_number(gym.num_trainable_weights),fmt_seconds(time.time()-t0)))
        
    t0               = time.time()
    training_info    = TrainingInfo( 
                                batch_size     = batch_size,
                                epochs         = epochs,
                                output_level   = output_level,
                                num_weights    = gym.num_trainable_weights)        
    monitor          = Monitor( gym            = gym, 
                                world          = world, 
                                val_world      = val_world,
                                result0        = result0, 
                                training_info  = training_info,
                                config         = config,
                                output_level   = output_level)
    if output_level != "quiet": print("Training monitor initialized. Took %s" % fmt_seconds(time.time()-t0))
    config.done()

    t0               = time.time()
    if monitor.is_done:
        why_stopped = "Cached model already sufficiently trained"
    else:
        assert epochs > (monitor.current_epoch+1), "Internal error. monitor.is_done failed"
        # tensorboard
        # See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html

        tboard = None
        if tboard_log_dir != "":
            t0             = time.time()
            tboard_log_dir = SubDir(tboard_log_dir).path
            tboard         = tf.keras.callbacks.TensorBoard(log_dir=tboard_log_dir, histogram_freq=tboard_freq, profile_batch=tboard_prf_batch )
            if output_level != "quiet": print("TensorBoard log directory set to '%s'. Took %s" % (tboard_log_dir, fmt_seconds(time.time()-t0)))

        why_stopped = "Training complete"
        try:
            gym.fit(        x              = world.tf_data,
                            y              = world.tf_y,
                            batch_size     = batch_size,
                            sample_weight  = world.tf_sample_weights * float(world.nSamples),  # sample_weights are poorly handled in TF
                            epochs         = epochs - (monitor.current_epoch+1),
                            callbacks      = monitor if tboard is None else [ monitor, tboard ],
                            verbose        = 0 )
        except KeyboardInterrupt:
            why_stopped = "Aborted"

    monitor.finalize(status = why_stopped)
    if output_level != "quiet": print("Training terminated. Total time taken %s" % fmt_seconds(time.time()-t0))





