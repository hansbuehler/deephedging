# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for 

June 30, 2022
@author: hansbuehler
"""

import numpy as np
import time as time
from cdxbasics.prettydict import PrettyDict as pdct
from cdxbasics.dynaplot import colors_tableau, figure
from .base import Logger, npCast, fmt_seconds, mean, err, tf, mean_bins, mean_cum_bins, perct_exp

_log = Logger(__file__)

colors = colors_tableau

# -------------------------------------------------------
# By epoch
# -------------------------------------------------------

class Plot_Loss_By_Epoch(object): # NOQA
    """
    Plot object for displaying learning progress
    """
    
    def __init__(self, *, fig, title, epochs, err_dev, lookback_window, show_epochs ): # NOQA
        
        self.lookback_window  = lookback_window
        self.show_epochs      = show_epochs
        self.show_epochs      = min( self.show_epochs, epochs)
        self.err_dev          = err_dev        
        self.lines            = {}
        self.fills            = {}
        self.line_best        = None            
        self.ax               = fig.add_subplot()
        self.ax.set_title(title)
        self.ax.set_xlim(1,self.show_epochs)
        self.ax.set_ylim(-0.1,+0.1)
        self.ax.set_xlabel("Epochs")
        
    def update(self, *, epoch, losses : dict, loss_errs : dict, best_epoch : int, best_loss : float ): # NOQA
            
        show_epoch0 = max( 0, epoch-self.show_epochs )
        show_epoch0 = min( best_epoch, show_epoch0 )
        first       = self.line_best is None
        
        _lines      = dict( full="-",batch="-" )
        _alphas     = dict( full=0.2, val=0.05 )
        
        x    = np.linspace(show_epoch0+1,epoch+1,epoch-show_epoch0+1,endpoint=True,dtype=np.int32 )
        min_ = None
        max_ = None
        for loss, color in zip( losses, colors() ):
            mean = np.array( losses[loss] )[show_epoch0:epoch+1]

            if self.lines.get(loss,None) is None:
                self.lines[loss] = self.ax.plot( x, mean, _lines.get(loss,":"), label=loss, color=color )[0]
            else:
                self.lines[loss].set_xdata( x )
                self.lines[loss].set_ydata( mean )

            # std error
            if loss in loss_errs:
                err  = np.array( loss_errs[loss] )[show_epoch0:epoch+1] * self.err_dev
                assert len(err.shape) == 1 and err.shape == mean.shape, "Error %s and %s" % (err.shape, mean.shape)
                if loss in self.fills:
                    self.fills[loss].remove()
                self.fills[loss] = self.ax.fill_between( x, y1=mean-err, y2=mean+err, color=color, alpha=_alphas[loss] )

            # y min/max            
            if loss != "init":
                min__ = min(mean[-self.lookback_window:])
                max__ = max(mean[-self.lookback_window:])
                if min_ is None or min_ > min__:
                    min_ = min__
                if max_ is None or max_ < max__:
                    max_ = max__
                    
        # indicator of best
        if self.line_best is None:
            self.line_best = self.ax.plot( [best_epoch+1], [best_loss], "*", label="best", color="black" )[0]
        else:
            self.line_best.set_xdata( [best_epoch+1] )
            self.line_best.set_ydata( [best_loss] )
            
        # adjust graph
        self.ax.set_xlim(show_epoch0+1,show_epoch0+1+self.show_epochs)
        self.ax.set_ylim(min_-0.001,max_+0.001)
        if first: self.ax.legend()

class Plot_Utility_By_Epoch(object): # NOQA
    """
    Plot object for displaying utilities
    """
    
    def __init__(self, *, fig, err_dev, epochs, lookback_window, show_epochs ): # NOQA
        
        self.lookback_window  = lookback_window
        self.show_epochs      = show_epochs
        self.show_epochs      = min( self.show_epochs, epochs)
        self.err_dev          = err_dev
        self.lines            = None
        self.fills            = None
        self.ax               = fig.add_subplot()
        self.ax.set_title("Monetay Utility")
        self.ax.set_xlabel("Epochs")
        
    def update(self, *, epoch, best_epoch, full_util, full_util0, full_util_err, full_util0_err, val_util, val_util0 ):# NOQA
    
        show_epoch0 = max( 0, epoch-self.show_epochs )
        show_epoch0 = min( best_epoch, show_epoch0 )
       
        x              = np.linspace(show_epoch0+1,epoch+1,epoch-show_epoch0+1,endpoint=True,dtype=np.int32 )
        best_full      = full_util[best_epoch]
        
        full_util      = np.array( full_util  )[show_epoch0:epoch+1]
        full_util0     = np.array( full_util0 )[show_epoch0:epoch+1]
        full_util_err  = np.array( full_util_err  )[show_epoch0:epoch+1]
        full_util0_err = np.array( full_util0_err )[show_epoch0:epoch+1]
        val_util       = np.array( val_util  )[show_epoch0:epoch+1]
        val_util0      = np.array( val_util0 )[show_epoch0:epoch+1]        
        
        if self.lines is None:
            self.lines = pdct()
            self.lines.full_util  = self.ax.plot( x, full_util,  "-", label="gains, full", color="red" )[0]
            self.lines.full_util0 = self.ax.plot( x, full_util0, "-", label="payoff, full", color="blue" )[0]
            self.lines.val_util   = self.ax.plot( x, val_util,   ":", label="gains, val", color="red" )[0]
            self.lines.val_util0  = self.ax.plot( x, val_util0,  ":", label="payoff, val", color="blue" )[0]
            self.lines.best       = self.ax.plot( [best_epoch+1], [best_full], "*", label="best", color="black" )[0]
            self.ax.legend()
            
        else:
            self.lines.full_util.set_ydata( full_util )
            self.lines.full_util0.set_ydata( full_util0 )
            self.lines.val_util.set_ydata( val_util )
            self.lines.val_util0.set_ydata( val_util0 )
            for line in self.lines:
                if line == 'best':
                    pass
                self.lines[line].set_xdata(x)
            self.lines.best.set_xdata( [best_epoch+1] )
            self.lines.best.set_ydata( [best_full] )

        if self.fills is None:            
            self.fills = pdct()
        else:
            for k in self.fills:
                self.fills[k].remove()
            self.fills.full_util  = self.ax.fill_between( x, full_util-full_util_err*self.err_dev, full_util+full_util_err*self.err_dev, color="red", alpha=0.2 )
            self.fills.full_util0 = self.ax.fill_between( x, full_util0-full_util0_err*self.err_dev, full_util0+full_util0_err*self.err_dev, color="blue", alpha=0.1 )

        # y min/max            
        min_ = min( np.min(full_util[-self.lookback_window:]), np.min(full_util0[-self.lookback_window:]) )
        max_ = max( np.max(full_util[-self.lookback_window:]), np.max(full_util0[-self.lookback_window:]) )
        self.ax.set_ylim(min_-0.001,max_+0.001)
        self.ax.set_xlim(show_epoch0+1,show_epoch0+1+self.show_epochs)

# -------------------------------------------------------
# By terminal outcome
# -------------------------------------------------------

class Plot_Returns_By_Percentile(object): # NOQA
    """
    Plot object for showing hedging performance
    """
    
    def __init__(self, *, fig, set_name, bins ): # NOQA
        
        self.bins  = bins
        self.line  = None
        self.ax    = fig.add_subplot()            
        self.ax.set_title("Cash Returns by Percentile\n(%s)" % set_name )
        self.ax.set_xlabel("Percentile")
        
    def update(self, *, P, gains, payoff ):# NOQA

        gains   = np.sort( gains )
        payoff  = np.sort( payoff )
        gains   = mean_bins( gains, bins=self.bins, weights=P )
        payoff  = mean_bins( payoff, bins=self.bins, weights=P )
        
        x = np.linspace( 0., 1., self.bins, endpoint=True )
        
        if self.line is None:
            self.line =  self.ax.plot( x, gains, label="gains" )[0]
            self.ax.plot( x, payoff, ":", label="payoff" )
            self.ax.plot( x, payoff*0., ":", color="black" )
            self.ax.legend()
            self.ax.set_xlim( 0.-0.1, 1.+0.1 )
        else:
            self.line.set_ydata( gains )

        min_ = min( np.min(gains), np.min(gains) )
        max_ = max( np.max(gains), np.max(gains) )
        if min_ >= max_-0.00001:
            min_ -= 0.0001
            max_ += 0.0001
        #self.ax.set_ylim(min_,max_)            

class Plot_Returns_By_Spot_Ret(object): # NOQA
    """
    Plot object for showing hedging performance
    """
    
    def __init__(self, *, fig, set_name, bins ): # NOQA
        
        self.bins    = bins
        self.ax      = fig.add_subplot()            
        self.line    = None
        self.line_h  = None
        self.ax.set_title("Cash Returns by Spot Return\n(%s))" % set_name )
        self.ax.set_xlabel("Spot return")
        
    def update(self, *, P, gains, hedge, payoff, spot_ret ):# NOQA

        ixs      = np.argsort( spot_ret )
        x        = spot_ret[ixs]
        gains    = gains[ixs]
        hedge     = hedge[ixs]
        payoff   = payoff[ixs]
        x        = mean_bins( x, bins=self.bins, weights=P )
        gains    = mean_bins( gains, bins=self.bins, weights=P )
        hedge    = mean_bins( hedge, bins=self.bins, weights=P )
        payoff   = mean_bins( payoff, bins=self.bins, weights=P )
        
        if self.line is None:
            self.line   =  self.ax.plot( x, gains, label="gains" )[0]
            self.ax.plot( x, payoff, ":", label="payoff" )
            self.line_h =  self.ax.plot( x, hedge, label="hedge" )[0]
            self.ax.plot( x, payoff*0., ":", color="black" )
            self.ax.legend()
        else:
            self.line.set_ydata( gains )
            self.line_h.set_ydata( hedge )

        min_ = min( np.min(gains), np.min(gains) )
        max_ = max( np.max(gains), np.max(gains) )
        if min_ >= max_-0.00001:
            min_ -= 0.0001
            max_ += 0.0001
        #self.ax.set_ylim(min_,max_)            

class Plot_Utility_By_CumPercentile(object): # NOQA
    """
    Plot object for showing hedging performance
    """
    
    def __init__(self, *, fig, set_name, bins ): # NOQA
        
        self.bins   = bins
        self.line   = None
        self.ax     = fig.add_subplot()            
        self.ax.set_title("Utility by cummlative percentile\n(%s)" % set_name)
        self.ax.set_xlabel("Percentile")
        
    def update(self, *, P, utility, utility0 ):# NOQA

        # percentiles
        # -----------
        bins     = min(self.bins, len(utility))      
        utility  = np.sort(utility)
        utility0 = np.sort(utility0)
        utility  = mean_cum_bins(utility, bins=self.bins, weights=P )
        utility0 = mean_cum_bins(utility0, bins=self.bins, weights=P  )
        x        = np.linspace(0.,1.,bins, endpoint=True)
        
        if self.line is None:
            self.line =  self.ax.plot( x, utility, label="gains" )[0]
            self.ax.plot( x, utility0, ":", label="payoff" )
            self.ax.plot( x, utility0*0., ":", color="black" )
            self.ax.legend()
            self.ax.set_xlim( 0.-0.1, 1.+0.1 )
        else:
            self.line.set_ydata( utility )

        min_ = min( np.min(utility), np.min(utility0) )
        max_ = max( np.max(utility), np.max(utility0) )
        if min_ >= max_-0.00001:
            min_ -= 0.0001
            max_ += 0.0001
        #self.ax.set_ylim(min_,max_)            

class Plot_Activity_By_Step(object): # NOQA
    """
    Plot object for showing hedging performance
    """
    
    def __init__(self, *, fig, activity_name, pcnt_lo, pcnt_hi, inst_names ): # NOQA
        
        self.pcnt_lo    = pcnt_lo
        self.pcnt_hi    = pcnt_hi
        self.inst_names = inst_names
        self.lines      = None
        self.fills      = None
        self.ax         = fig.add_subplot()
        self.ax.set_title("Activity by time step\n(%s)" % activity_name)
        self.ax.set_xlabel("Step")
        
    def update(self, *, P, action ):# NOQA

        nSamples = action.shape[0]
        nSteps   = action.shape[1]
        nInst    = action.shape[2]
        assert nInst == len(self.inst_names), "Internal error: %ld != %ld" % ( nInst, len(self.inst_names) )
        first    = self.lines is None
        
        # percentiles
        # -----------
        
        x = np.linspace(1,nSteps,nSteps,endpoint=1,dtype=np.int32)

        if self.lines is None:
            self.lines = []
            self.fills = []

        for iInst, color in zip( range(nInst), colors() ):
            action_i       = action[:,:,iInst]
            percentiles_i  = perct_exp( action_i, lo=self.pcnt_lo, hi=self.pcnt_hi, weights=P )
            mean_i         = np.sum( action_i*P[:,np.newaxis], axis=0, ) / np.sum(P)
            assert percentiles_i.shape[1] == 2, "error %s" % percentiles_i.shape

            if first:
                self.lines.append( self.ax.plot( x, mean_i, "-", color=color, label=self.inst_names[iInst] )[0] )
                self.fills.append( None )
            else:
                self.lines[iInst].set_ydata( mean_i )
                self.fills[iInst].remove()
            self.fills[iInst] = self.ax.fill_between( x, percentiles_i[:,0], percentiles_i[:,1], color=color, alpha=0.2 ) 

        if first:
            self.ax.legend()

# -------------------------------------------------------
# Monitor
# -------------------------------------------------------

class NotebookMonitor(tf.keras.callbacks.Callback):
    """
    Monitors progress of our training and displays the result
    in a jupyter notebook with dynamic graphing.    
    
    "Base class" is trainer.NoMonitor
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
        
        print("\rKDeep Hedging Engine: warming up for %ld epochs and %ld samples (%ld validation samples) ...         " % (epochs, world.nSamples, self.val_world.nSamples), end='')

    def on_epoch_begin( self, epoch, logs = None ):# NOQA
        if self.epoch == -1:
            print("\r\33[2KDeep Hedging Engine: first of %ld epochs for %ld samples (%ld validation samples) started. Compiling graph ...       " % (self.epochs, self.world.nSamples, self.val_world.nSamples), end='')
            
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
            print("\r")
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
        print("\r%s | %s | %s                         " % ( str_intro, str_perf, str_time ), end='')
              
    def finalize( self, set_best = True ):
        """ Plot final result and copy best weights to model """
        if self.why_stopped == "Aborted":
            print("\r                                      \r*** Aborted ... ", end='')
        self.plot()
        self.gym.set_weights( self.best_weights )
        print("\n Status: %s\n" % self.why_stopped )