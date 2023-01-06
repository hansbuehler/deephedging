# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for 

June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, Int
from .plot_training import NotebookMonitor
_log = Logger(__file__)

class NoMonitor(tf.keras.callbacks.Callback):
    """ Does nothing. Sort of base class for Monitors """
    
    def __init__(self, gym, world, val_world, result0, epochs, batch_size, time_out, config ):# NOQA
        tf.keras.callbacks.Callback.__init__(self)
        self.epochs           = epochs
        self.why_stopped      = "Ran all %ld epochs" % epochs

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

def MonitorFactory( gym, world, val_world, result0, epochs, batch_size, time_out, config ):
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
        config    : config

    Returns
    -------
        An monitor.
    """    
    monitor_type  = config("monitor_type", "notebook", str, "What kind of monitor to instantiate")
    monitor       = None
    if monitor_type == "notebook":
        monitor = NotebookMonitor( gym, world, val_world, result0, epochs, batch_size, time_out, config  )
    elif monitor_type == "none":
        monitor = NoMonitor( gym, world, val_world, result0, epochs, batch_size, time_out, config  )
    
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
    """
    #tf.debugging.enable_check_numerics()
    
    optimzier        = config.train("optimizer", "RMSprop", help="Optimizer" )
    batch_size       = config.train("batch_size", None, help="Batch size")
    epochs           = config.train("epochs", 100, Int>0, help="Epochs")
    time_out         = config.train("time_out", None, int, help="Timeout in seconds. None for no timeout.")
    run_eagerly      = config.train("run_eagerly", False, help="Keras model run_eagerly. Turn to True for debugging. This slows down training. Use None for default.")
    learning_rate    = config.train("learing_rate", None, help="Manually set the learning rate of the optimizer")
    tboard_log_dir   = config.train.tensor_board("log_dir", "", str, "Specify tensor board log directory")
    tboard_freq      = config.train.tensor_board("hist_freq", 1, Int>0, "Specify tensor board log frequency")

    result0          = gym(world.tf_data)
    monitor          = MonitorFactory(  gym=gym, 
                                        world=world, 
                                        val_world=val_world,
                                        result0=result0, 
                                        epochs=epochs, 
                                        batch_size=batch_size, 
                                        time_out=time_out,
                                        config=config.visual )
 
    # See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html
    tboard = None
    if tboard_log_dir != "":
        tboard        = tf.keras.callbacks.TensorBoard(log_dir=tboard_log_dir, histogram_freq=tboard_freq)
        print("\r\33[2KTensorBoard log directory set to %s" % tboard_log_dir)
                                                 
    gym.compile(    optimizer        = optimzier, 
                    loss             = dict( loss=default_loss ),
                    weighted_metrics = dict( loss=default_loss ),
                    run_eagerly      = run_eagerly)

    if not learning_rate is None:
        gym.optimizer.lr = float( learning_rate )

    try:
        gym.fit(        x              = world.tf_data,
                        y              = world.tf_y,
                        batch_size     = batch_size,
                        sample_weight  = world.tf_sample_weights * float(world.nSamples),  # sample_weights are poorly handled in TF
                        epochs         = epochs,
                        callbacks      = monitor if tboard_log_dir != "" else [ monitor, tboard ],
                        verbose        = 0 )
    except KeyboardInterrupt:
        monitor.why_stopped = "Aborted"
    monitor.finalize()







