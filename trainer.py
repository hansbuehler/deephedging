# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for 

June 30, 2022
@author: hansbuehler
"""

import time as time
import numpy as np
from cdxbasics.prettydict import PrettyDict as pdct
from cdxbasics.dynaplot import figure
from .base import Logger, Config, tf, npCast, fmt_seconds, mean, err
from .plot_training import Plot_Loss_By_Epoch, Plot_Utility_By_Epoch, Plot_Returns_By_Percentile, Plot_Returns_By_Spot_Ret, Plot_Utility_By_CumPercentile, Plot_Activity_By_Step
_log = Logger(__file__)

class NotebookMonitor(tf.keras.callbacks.Callback):
    """
    Monitors progress of our training and displays the result
    in a jupyter notebook with dynamic graphing.    
    """
    
    def __init__(self, gym, world, val_world, result0, epochs, batch_size, time_out, config ):# NOQA
        tf.keras.callbacks.Callback.__init__(self)
        self.time_refresh     = config("time_refresh", 20, int, help="Time refresh interval for visualizations" )
        self.epoch_refresh    = config("epoch_refresh", 10, int, help="Epoch fefresh frequency for visualizations" )        
        self.fig_row_size     = config.fig("row_size", 5, int, "Plot size of a row")
        self.fig_col_size     = config.fig("col_size", 5, int, "Plot size of a column")
        self.fig_col_nums     = config.fig("col_nums", 6, int, "Number of columbs")
        self.err_dev          = config("err_dev", 1., float, help="How many standard errors to add to loss to assess best performance" )   
        self.lookback_window  = config("lookback_window", 30, int, "Lookback window for determining y min/max")
        self.show_epochs      = config("show_epochs", 100, int,  "Maximum epochs displayed")
        self.bins             = config("bins", 200, int, "How many x to plot")
        self.pcnt_lo          = config("confidence_pcnt_lo", 0.5, float, "Lower percentile for confidence intervals")
        self.pcnt_hi          = config("confidence_pcnt_hi", 0.5, float, "Upper percentile for confidence intervals")
        
        self.gym              = gym
        self.world            = world
        self.val_world        = val_world
        self.result0          = result0
        self.P                = world.sample_weights
        self.val_P            = val_world.sample_weights
        self.started          = False
        self.epochs           = epochs
        self.time_out         = time_out
        self.time0            = time.time()
        self.time_last        = -1
        self.batch_size       = batch_size if not batch_size is None else 32
        self.epoch            = -1
        self.why_stopped      = "Ran all %ld epochs" % epochs
        self.fig              = None
        
        # track progress
        self.losses            = pdct()
        self.losses.init       = []
        self.losses.full       = []
        self.losses.batch      = []
        self.losses.val        = []
        self.losses_err        = pdct()
        self.losses_err.full   = []
        self.losses_err.val    = []

        self.init_loss         = mean( self.P, result0.loss )
        self.init_loss_err     = err( self.P, result0.loss )
        self.best_loss         = self.init_loss 
        self.best_loss_err     = self.init_loss_err
        self.best_weights      = self.gym.get_weights()
        self.best_epoch        = 0        
        
        self.utilities         = pdct()
        self.utilities.full_util      = []
        self.utilities.full_util0     = []
        self.utilities.full_util_err  = []
        self.utilities.full_util0_err = []
        self.utilities.val_util       = []
        self.utilities.val_util0      = []
        
        print("\r\33[2KDeep Hedging Engine: warming up for %ld epochs and %ld samples / %ld validation samples ... " % (epochs, world.nSamples, self.val_world.nSamples), end='')

    def on_epoch_begin( self, epoch, logs = None ):# NOQA
        if self.epoch == -1:
            print("\r\33[2KDeep Hedging Engine: first of %ld epochs for %ld samples / %ld validation samples started ... " % (self.epochs, self.world.nSamples, self.val_world.nSamples), end='')
            
    def on_epoch_end( self, epoch, logs = None ):
        """ Called when an epoch ends """
        self.full_result = npCast( self.gym(self.world.tf_data) )
        self.val_result  = npCast( self.gym(self.val_world.tf_data) )
        self.epoch       = epoch

        # losses
        # Note that we apply world.sample_weights to all calculations
        # so we are in sync with keras.fit()
        self.losses.batch.append(   float( logs['loss_default_loss'] ) ) # we read the metric instead of 'loss' as this appear to be weighted properly
        self.losses.full.append(    mean(self.P, self.full_result.loss) )
        self.losses.val.append(     mean(self.val_P, self.val_result.loss) )
        self.losses.init.append(    self.init_loss )

        self.losses_err.full.append( err(self.P,self.full_result.loss) )
        self.losses_err.val.append(  err(self.val_P,self.val_result.loss) )
        
        if self.losses.full[-1] < self.best_loss :
            self.best_loss         = self.losses.full[-1]
            self.best_loss_err     = self.best_loss_err
            self.best_weights      = self.gym.get_weights()
            self.best_epoch        = epoch
            
        # utilities
        self.utilities.full_util.append(     mean(self.P, self.full_result.utility ) )
        self.utilities.full_util0.append(    mean(self.P, self.full_result.utility0) )
        self.utilities.full_util_err.append( err( self.P, self.full_result.utility ) )
        self.utilities.full_util0_err.append(err( self.P, self.full_result.utility0) )
        self.utilities.val_util.append(      mean(self.val_P, self.val_result.utility ) )
        self.utilities.val_util0.append(     mean(self.val_P, self.val_result.utility0 ) )
        
        # print or not
        # ------------
        
        total_time_passed = time.time() - self.time0 
        if not self.time_out is None and total_time_passed > self.time_out:
            print("\r\33[2K**** Training timed out after a run time of %s" % fmt_seconds(total_time_passed))
            self.model.stop_training = True
            self.why_stopped         = "Timeout after %s" % fmt_seconds(total_time_passed)
        elif epoch != 0 and \
            epoch % self.epoch_refresh != 0 and \
            ( time.time() - self.time_last ) < self.time_refresh:
            return
        
        self.plot()
        
    def plot(self):
        """ 
        Update our plots
        Create figures and subplots if not done so before
        """
        assert self.epoch >= 0, "Do not call me before the first epoch"
        
        if self.fig is None:
            print("\r\33[2", end='')  # clear any existing messages
            # create figure
            self.fig                        = figure(row_size=self.fig_row_size, col_size=self.fig_col_size, col_nums=self.fig_col_nums, tight=True )
            
            # by epoch
            self.plot_loss_by_epoch         = Plot_Loss_By_Epoch(    fig=self.fig, title="Losses (recent)", epochs=self.epochs, err_dev=self.err_dev, lookback_window=self.lookback_window, show_epochs=self.show_epochs )
            self.plot_loss_by_epoch_all     = Plot_Loss_By_Epoch(    fig=self.fig, title="Losses (all)", epochs=self.epochs, err_dev=self.err_dev, lookback_window=self.epochs, show_epochs=self.epochs )
            self.plot_utility_by_epoch      = Plot_Utility_By_Epoch( fig=self.fig, err_dev=self.err_dev, epochs=self.epochs, lookback_window=self.lookback_window, show_epochs=self.show_epochs )

            self.fig.next_row()
            
            # by performance - training
            self.plot_returns_by_spot_ret          = Plot_Returns_By_Spot_Ret(      fig=self.fig, set_name="training set", bins=self.bins )
            self.plot_returns_by_percentile        = Plot_Returns_By_Percentile(    fig=self.fig, set_name="training set", bins=self.bins )
            self.plot_utility_by_cumpercentile     = Plot_Utility_By_CumPercentile( fig=self.fig, set_name="training set", bins=self.bins )
            
            # by performance - validation
            self.val_plot_returns_by_spot_ret      = Plot_Returns_By_Spot_Ret(      fig=self.fig, set_name="validation set", bins=self.bins )
            self.val_plot_returns_by_percentile    = Plot_Returns_By_Percentile(    fig=self.fig, set_name="validation set", bins=self.bins )
            self.val_plot_utility_by_cumpercentile = Plot_Utility_By_CumPercentile( fig=self.fig, set_name="validation set", bins=self.bins )

            self.fig.next_row()

            # activity by step
            self.plot_actions_by_step          = Plot_Activity_By_Step(    fig=self.fig, activity_name="Actions\n(training set)", pcnt_lo=self.pcnt_lo, pcnt_hi=self.pcnt_hi, inst_names=self.world.inst_names )
            self.plot_deltas_by_step           = Plot_Activity_By_Step(    fig=self.fig, activity_name="Delta\n(training set)", pcnt_lo=self.pcnt_lo, pcnt_hi=self.pcnt_hi, inst_names=self.world.inst_names )

            self.fig.render()
        
        # update live graphics
        # --------------------
        self.fig.suptitle("Learning to Trade, epoch %ld / %ld" % (self.epoch+1,self.epochs), fontsize=20)
        
        # by epoch
        self.plot_loss_by_epoch.update( epoch=self.epoch, losses=self.losses, loss_errs=self.losses_err, best_epoch=self.best_epoch, best_loss=self.best_loss )
        self.plot_loss_by_epoch_all.update( epoch=self.epoch, losses=self.losses, loss_errs=self.losses_err, best_epoch=self.best_epoch, best_loss=self.best_loss )
        self.plot_utility_by_epoch.update( epoch=self.epoch, best_epoch=self.best_epoch, **self.utilities )

        # by performance - training
        adjusted_full_gains  = self.full_result.gains - mean(self.P, self.full_result.utility)
        adjusted_full_hedge  = (self.full_result.gains - self.full_result.payoff) - mean(self.P, self.full_result.utility)
        adjusted_full_payoff = self.full_result.payoff - mean(self.P, self.full_result.utility0)
        self.plot_returns_by_spot_ret.update( P=self.P, gains=adjusted_full_gains, hedge=adjusted_full_hedge, payoff=adjusted_full_payoff, spot_ret=self.world.diagnostics.per_path.spot_ret )
        self.plot_returns_by_percentile.update( P=self.P, gains=adjusted_full_gains, payoff=adjusted_full_payoff )
        self.plot_utility_by_cumpercentile.update( P=self.P, utility=self.full_result.utility, utility0=self.full_result.utility0 )
        
        # by performance - validation
        adjusted_val_gains  = self.val_result.gains - mean(self.val_P, self.val_result.utility)
        adjusted_val_hedge  = (self.val_result.gains - self.val_result.payoff) - mean(self.val_P, self.val_result.utility)
        adjusted_val_payoff = self.val_result.payoff - mean(self.val_P, self.val_result.utility0)
        self.val_plot_returns_by_spot_ret.update( P=self.val_P, gains=adjusted_val_gains, hedge=adjusted_val_hedge, payoff=adjusted_val_payoff, spot_ret=self.val_world.diagnostics.per_path.spot_ret )
        self.val_plot_returns_by_percentile.update( P=self.val_P, gains=adjusted_val_gains, payoff=adjusted_val_payoff )
        self.val_plot_utility_by_cumpercentile.update( P=self.val_P, utility=self.val_result.utility, utility0=self.val_result.utility0 )
        
        # activity by step
        assert len(self.full_result.actions.shape) == 3, "Shape %s" % self.full_result.actions.shape
        self.plot_actions_by_step.update( P=self.P, action=self.full_result.actions )
        self.plot_deltas_by_step.update( P=self.P, action=self.full_result.deltas )
        self.fig.render()

        # Print
        # -----        
        full_loss_mean    = self.losses.full[-1]
        full_loss_err     = self.losses_err.full[-1]
        val_loss_mean     = self.losses.val[-1]
        val_loss_err      = self.losses_err.val[-1]
        batch_loss        = self.losses.batch[-1]
        total_time_passed = time.time() - self.time0 
        time_left         = total_time_passed / float(self.epoch+1) * float(self.epochs-self.epoch)
        
        str_intro = "Training %ld/%ld epochs; %ld samples; %ld validation samples batch size %ld" % ( self.epoch+1, self.epochs, self.world.nSamples, self.val_world.nSamples, self.batch_size)
        str_perf  = "initial loss %g (%g), full %g (%g), best %g (%g), batch %g, val %g (%g). Best batch %ld" % ( \
                                        self.init_loss, self.init_loss_err, \
                                        full_loss_mean, full_loss_err, \
                                        self.best_loss, self.best_loss_err, \
                                        batch_loss, \
                                        val_loss_mean, val_loss_err, \
                                        self.best_epoch )
        str_time  = "time elapsed %s; estimated time remaining %s" % ( fmt_seconds(total_time_passed), fmt_seconds(time_left) )        
        str_time  = str_time if self.time_out is None else str_time + ("; time out %ld" % fmt_seconds(self.time_out))
        print("\r\33[2K%s | %s | %s  " % ( str_intro, str_perf, str_time ), end='')
              
    def finalize( self, set_best = True ):
        """ Plot final result and copy best weights to model """
        if self.why_stopped == "Aborted":
            print("\r\33[2K*** Aborted ...", end='')
        self.plot()
        self.gym.set_weights( self.best_weights )
        print("\n Status: %s\n" % self.why_stopped )


class NoMonitor(tf.keras.callbacks.Callback):
    """ Does nothing """
    
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
        result0   : gym(world)
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
        verbose   : how much detail to print

    Returns
    -------
        Nothing.        
            Run gym(world) to get trained data for the full data set.
            The gym itself contains the weights of the trained agents.
    """
    #tf.debugging.enable_check_numerics()
    
    optimzier        = config.train("optimizer", "adam", help="Optimizer" )
    batch_size       = config.train("batch_size", None, help="Batch size")
    epochs           = config.train("epochs", 100, int, help="Epochs")
    time_out         = config.train("time_out", None, int, help="Timeout in seconds. None for no timeout.")
    run_eagerly      = config.train("run_eagerly", False, bool, "Keras model run_eagerly")
    result0          = gym(world.tf_data)
    monitor          = MonitorFactory(  gym=gym, 
                                        world=world, 
                                        val_world=val_world,
                                        result0=result0, 
                                        epochs=epochs, 
                                        batch_size=batch_size, 
                                        time_out=time_out,
                                        config=config.visual )
 
    gym.compile(    optimizer        = optimzier, 
                    loss             = dict( loss=default_loss ),
                    weighted_metrics = dict( loss=default_loss ),
                    run_eagerly = run_eagerly)

    try:
        gym.fit(        x              = world.tf_data,
                        y              = world.tf_y,
                        batch_size     = batch_size,
                        sample_weight  = world.tf_sample_weights * float(world.nSamples),  # sample_weights are poorly handled in TF
                        epochs         = epochs,
                        callbacks      = monitor,
                        verbose        = 0 )
    except KeyboardInterrupt:
        monitor.why_stopped = "Aborted"
    monitor.finalize()







