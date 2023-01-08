# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for 

June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, Int, Float
from .plot_training import NotebookMonitor
_log = Logger(__file__)

class NoMonitor(tf.keras.callbacks.Callback):
    """ Does nothing. Sort of base class for Monitors """
    
    def __init__(self, gym, world, val_world, result0, epochs, batch_size, time_out, config_visual, config_caching ):# NOQA
        tf.keras.callbacks.Callback.__init__(self)
        self.epochs           = epochs
        self.epoch            = -1
        self.why_stopped      = "Ran all %ld epochs" % epochs
        
    @property
    def is_done(self):
        return self.epoch+1 >= self.epochs

    def on_epoch_begin( self, epoch, logs = None ):# NOQA
        pass
            
    def on_epoch_end( self, epoch, logs = None ):
        """ Called when an epoch ends """
        self.epoch       = epoch
        
    def plot(self):# NOQA
        pass
              
    def finalize( self, set_best = True ):# NOQA
        pass

# =========================================================================================
# Factory
# =========================================================================================

def MonitorFactory( gym, world, val_world, result0, epochs, batch_size, time_out, monitor_type, config_visual, config_caching ):
    """
    Creates a monitor based on a config file. A monitor prints progress information during training.

    Parameters
    ----------
        gym       : VanillaDeepHedgingGym or similar interface
        world     : world with training data
        val_world : world with validation data (e.g. computed using world.clone())
        result0   : result from gym(world)
        epochs    : number of epochs
        batch_size: batch_size
        time_out  : in seconds
        config_visual,
        config_caching : configs for visualization and caching, respectively.

    Returns
    -------
        An monitor.
    """    
    monitor       = None
    if monitor_type == "notebook":
        monitor = NotebookMonitor( gym, world, val_world, result0, epochs, batch_size, time_out, config_visual, config_caching )
    elif monitor_type == "none":
        monitor = NoMonitor( gym, world, val_world, result0, epochs, batch_size, time_out, config_visual, config_caching )
    
    _log.verify( not monitor is None, "Unknnown monitor type '%s'", monitor_type )
    return monitor

# =========================================================================================
# training
# =========================================================================================
    
def default_loss( y_true,y_pred ):     
    """ Default loss: ignore y_true """
    return y_pred

def train(  gym,
            world,
            val_world,
            config  : Config,
            verbose : int =0):
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
        verbose   : how much detail to print during training (not used)

    Returns
    -------
        Nothing.        
            Run gym(world) to get trained data for the full data set.
            The gym itself contains the weights of the trained agents.
            
    The current training loop set up is a bit messy between managing user feedback, and also allowing to cache results
    to support warm starting. Will at some point redesign this architecture to create cleaner delineation of data, caching, 
    and visualization (at the very least to support multi-processing training)
    """
    tf.debugging.disable_check_numerics()
    
    optimzier        = config.train("optimizer",  "RMSprop", help="Optimizer" )
    batch_size       = config.train("batch_size",  None, help="Batch size")
    epochs           = config.train("epochs",      100, Int>0, help="Epochs")
    time_out         = config.train("time_out",    None, int, help="Timeout in seconds. None for no timeout.")
    run_eagerly      = config.train("run_eagerly", False, help="Keras model run_eagerly. Turn to True for debugging. This slows down training. Use None for default.")
    learning_rate    = config.train("learing_rate", None, help="Manually set the learning rate of the optimizer")
    
    # monitoring and caching
    monitor_type     = config("monitor_type", "notebook", ['notebook', 'none'], "What kind of progress monitor to use. Set to 'notebook' for jupyter use.")
    config_visual    = config.visual.detach()
    config_caching   = config.caching.detach()
    
    # tensorboard: have not been able to use it .. good luck.
    tboard_log_dir   = config.train.tensor_board(   "log_dir", "", str, "Specify tensor board log directory")
    tboard_freq      = config.train.tensor_board(   "hist_freq", 1, Int>0, "Specify tensor board log frequency")
    config.done()
    
    result0          = gym(world.tf_data)   # builds the model

    gym.compile(    optimizer        = optimzier, 
                    loss             = dict( loss=default_loss ),
                    weighted_metrics = dict( loss=default_loss ),
                    run_eagerly      = run_eagerly)

    if not learning_rate is None:
        gym.optimizer.lr = float( learning_rate )

    monitor          = MonitorFactory(  gym            = gym, 
                                        world          = world, 
                                        val_world      = val_world,
                                        result0        = result0, 
                                        epochs         = epochs, 
                                        batch_size     = batch_size, 
                                        time_out       = time_out,
                                        monitor_type   = monitor_type,
                                        config_visual  = config_visual,
                                        config_caching = config_caching )
    
    if monitor.is_done:
        monitor.why_stopped = "Cached model already sufficiently trained"
    else:
        # tensorboard
        # See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html
        tboard = None
        if tboard_log_dir != "":
            tboard        = tf.keras.callbacks.TensorBoard(log_dir=tboard_log_dir, histogram_freq=tboard_freq)
            print("\r\33[2KTensorBoard log directory set to %s" % tboard_log_dir)

        try:
            gym.fit(        x              = world.tf_data,
                            y              = world.tf_y,
                            batch_size     = batch_size,
                            sample_weight  = world.tf_sample_weights * float(world.nSamples),  # sample_weights are poorly handled in TF
                            epochs         = epochs - monitor.cache_epoch_off,
                            callbacks      = monitor if tboard is None else [ monitor, tboard ],
                            verbose        = 0 )
        except KeyboardInterrupt:
            monitor.why_stopped = "Aborted"

    monitor.finalize()







