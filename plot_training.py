# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for 

June 30, 2022
@author: hansbuehler
"""

import numpy as np
import psutil as psutil
from datetime import datetime, timedelta
from cdxbasics.prettydict import PrettyDict as pdct
from cdxbasics.dynaplot import colors_tableau, figure
from cdxbasics.config import Config
from .base import Logger, fmt_seconds, mean, mean_bins, mean_cum_bins, perct_exp, Int, Float, fmt_big_number, fmt_now, fmt_datetime

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
        """
        Parameters
        ----------
            fig 
            title 
            epochs  : total epochs
            err_dev : error bar
            loookback_window : look back for computing y axis
            show_epochs : how many epochs to show, at most
        """
        
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
        self._lines           = dict( training="-",batch="-" )
        self._alphas          = dict( training=0.2, val=0.05 )
        
    def update(self, *, epoch, losses : dict, loss_errs : dict, best_epoch : int, best_loss : float ): # NOQA
            
        show_epoch0 = max( 0, epoch-self.show_epochs )
        first       = self.line_best is None        
        x           = np.linspace(show_epoch0+1,epoch+1,epoch-show_epoch0+1,endpoint=True,dtype=np.int32 )
        min_        = None
        max_        = None
        for loss, color in zip( losses, colors() ):
            losses_here = np.array( losses[loss] )[show_epoch0:epoch+1]

            if self.lines.get(loss,None) is None:
                self.lines[loss] = self.ax.plot( x, losses_here, self._lines.get(loss,":"), label=loss, color=color )[0]
            else:
                self.lines[loss].set_xdata( x )
                self.lines[loss].set_ydata( losses_here )

            # std error
            if loss in loss_errs:
                err_  = np.array( loss_errs[loss] )[show_epoch0:epoch+1] * self.err_dev
                assert len(err_.shape) == 1 and err_.shape == losses_here.shape, "Error %s and %s" % (err_.shape, losses_here.shape)
                if loss in self.fills:
                    self.fills[loss].remove()
                self.fills[loss] = self.ax.fill_between( x, y1=losses_here-err_, y2=losses_here+err_, color=color, alpha=self._alphas[loss] )
                min__ = min( (losses_here-err_)[-self.lookback_window:])
                max__ = max( (losses_here+err_)[-self.lookback_window:])
            else:
                min__ = min( losses_here[-self.lookback_window:])
                max__ = max( losses_here[-self.lookback_window:])

            # y min/max            
            if loss != "init":
                min_ = min__ if min_ is None else min( min_, min__ )
                max_ = max__ if max_ is None else max( max_, max__ )
                    
        # indicator of best
        if self.line_best is None:
            self.line_best = self.ax.plot( [max(best_epoch+1,x[0])], [best_loss], "*", label="best", color="black" )[0]
        else:
            self.line_best.set_xdata( [max(best_epoch+1,x[0])] )
            self.line_best.set_ydata( [best_loss] )
            
        # adjust graph
        self.ax.set_xlim(x[0] if x[0]<x[-1] else x[-1]-1,x[-1])
        dx   = max( max_-min_, 0.0001)
        max_ += dx/20.
        min_ -= dx/20.
        self.ax.set_ylim(min_,max_)
        if first: self.ax.legend()

class Plot_Utility_By_Epoch(object): # NOQA
    """
    Plot object for displaying utilities progress by epoch
    """
    
    def __init__(self, *, fig, name, label, err_dev, epochs, lookback_window, show_epochs ): # NOQA
        
        self.lookback_window  = lookback_window
        self.show_epochs      = show_epochs
        self.show_epochs      = min( self.show_epochs, epochs)
        self.err_dev          = err_dev
        self.label            = label
        self.lines            = None
        self.fills            = pdct()
        self.ax               = fig.add_subplot()
        self.ax.set_title("%s Monetay Utility" % name)
        self.ax.set_xlabel("Epochs")
        
    def update(self, *, epoch, best_epoch, training_util, training_util_err, val_util ):# NOQA
    
        show_epoch0 = max( 0, epoch-self.show_epochs )
        x                  = np.linspace(show_epoch0+1,epoch+1,epoch-show_epoch0+1,endpoint=True,dtype=np.int32 )
        best_training      = training_util[best_epoch]
        training_util      = np.array( training_util  )[show_epoch0:epoch+1]
        training_util_err  = np.array( training_util_err  )[show_epoch0:epoch+1]
        val_util           = np.array( val_util  )[show_epoch0:epoch+1]
        
        if self.lines is None:
            self.lines = pdct()
            self.lines.training_util  = self.ax.plot( x, training_util,  "-", label="%s, training" % self.label, color="red" )[0]
            self.lines.val_util       = self.ax.plot( x, val_util,   ":",     label="%s, val" % self.label, color="red" )[0]
            self.lines.best           = self.ax.plot( [max(best_epoch+1,x[0])], [best_training], "*", label="best training", color="black" )[0]
            self.ax.legend()
            
        else:
            self.lines.training_util.set_ydata( training_util )
            self.lines.val_util.set_ydata( val_util )
            for line in self.lines:
                if line == 'best':
                    pass
                self.lines[line].set_xdata(x)
            self.lines.best.set_xdata( [max(best_epoch+1,x[0])] )
            self.lines.best.set_ydata( [best_training] )

        for k in self.fills:
            self.fills[k].remove()
        self.fills.training_util  = self.ax.fill_between( x, training_util-training_util_err*self.err_dev, training_util+training_util_err*self.err_dev, color="red", alpha=0.2 )

        # xy min/max            
        self.ax.set_xlim(x[0] if x[0]<x[-1] else x[-1]-1,x[-1])
        min_ = np.min(training_util[-self.lookback_window:]-training_util_err[-self.lookback_window:]*self.err_dev)
        max_ = np.max(training_util[-self.lookback_window:]+training_util_err[-self.lookback_window:]*self.err_dev)
        dx   = max( max_-min_, 0.0001)
        max_ += dx/20.
        min_ -= dx/20.
        self.ax.set_ylim(min_,max_)

class Plot_Memory_By_Epoch(object): # NOQA
    """
    Show memory usage
    """
    
    def __init__(self, *, fig, epochs ): # NOQA
        
        self._min             = None
        self._max             = None
        self.ax               = fig.add_subplot()
        self.ax.set_title("Memory usage by epoch")
        self.ax.set_xlim(0,epochs+1)
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Memory (GB)")
        self._x                = np.linspace(0,epochs,epochs+1,endpoint=True,dtype=np.int32)
        
    def update(self, *, epoch, process_info ): # NOQA
            
        memory_rss  = process_info.memory_rss
        memory_vms  = process_info.memory_vms

        first = self._min is None
        l     = len(process_info.memory_rss)
        if l > len(self._x):
            # this can happen if a cached objkect was trained for more epochs than currently requested
            self._x                = np.linspace(0,l-1,l,endpoint=True,dtype=np.int32)
            self.ax.set_xlim(0,l)
            
        if first:
            self.line_rss = self.ax.plot( self._x[:l], memory_rss, label="rss", color="blue" )[0]
            self.line_vms = self.ax.plot( self._x[:l], memory_vms, label="vms", color="green" )[0]
            self.ax.legend()
            
            self._min     = min( np.min(memory_rss), np.min(memory_vms) )
            self._max     = max( np.max(memory_rss), np.max(memory_vms) )
        else:
            self.line_rss.set_xdata( self._x[:l] )
            self.line_rss.set_ydata( memory_rss )
            self.line_vms.set_xdata( self._x[:l] )
            self.line_vms.set_ydata( memory_vms )
            
            self._min     = min( self._min, np.min(memory_rss), np.min(memory_vms) )
            self._max     = max( self._max, np.max(memory_rss), np.max(memory_vms) )

        self.ax.set_ylim(self._min-10.,self._max+10.)
        
# -------------------------------------------------------
# By terminal outcome
# -------------------------------------------------------

color_gains = "blue"
color_hedge = "green"
color_payoff = "orange"

class Plot_Returns_By_Spot_Ret(object): # NOQA
    """
    Plot object for showing hedging performance by return of spot (the most intuitive)
    """
    
    def __init__(self, *, fig, title, bins, with_std ): # NOQA
        self.bins     = bins
        self.with_std = with_std
        self.ax       = fig.add_subplot()            
        self.line     = None
        self.line_h   = None
        self.fills    = {}
        self.ax.set_title(title)
        self.ax.set_xlabel("Spot return")
        
    def update(self, *, P, gains, hedge, payoff, spot_ret ):# NOQA

        ixs      = np.argsort( spot_ret )
        x        = spot_ret[ixs]
        gains    = gains[ixs]
        hedge    = hedge[ixs]
        payoff   = payoff[ixs]
        x                  = mean_bins( x, bins=self.bins, weights=P, return_std=False )
        gains, gains_std   = mean_bins( gains, bins=self.bins, weights=P, return_std=True )
        hedge, hedge_std   = mean_bins( hedge, bins=self.bins, weights=P, return_std=True )
        payoff, payoff_std = mean_bins( payoff, bins=self.bins, weights=P, return_std=True )
        
        if self.line is None:
            self.line   =  self.ax.plot( x, gains, label="gains", color=color_gains )[0]
            self.ax.plot( x, payoff, ":", label="payoff", color=color_payoff )
            self.line_h =  self.ax.plot( x, -hedge, label="-hedge", color=color_hedge )[0]
            self.ax.plot( x, payoff*0., ":", color="black" )
            self.ax.legend()
        else:
            self.line.set_ydata( gains )
            self.line_h.set_ydata( -hedge )

        if self.with_std:
            for k in self.fills:
                self.fills[k].remove()
            self.fills['gains']   = self.ax.fill_between( x, gains-gains_std, gains+gains_std, color=color_gains, alpha=0.2 )
            self.fills['payoff']  = self.ax.fill_between( x, payoff-payoff_std, payoff+payoff_std, color=color_payoff, alpha=0.2 )
            self.fills['hedge']   = self.ax.fill_between( x, -hedge-hedge_std, -hedge+hedge_std, color=color_hedge, alpha=0.2 )

        # below is pretty heuristic. If training works, below makes sense
        # as the gains process and the payoff are similar.
        # it produces less good visuals if training is bad ...
        min_ = min( np.min(gains), np.min(payoff) )
        max_ = max( np.max(gains), np.max(payoff) )
        dx   = max( max_ - min_, 0.0001 )
        min_ -= dx/20.
        max_ += dx/20.
        self.ax.set_ylim(min_,max_)            

class Plot_Utility_By_CumPercentile(object): # NOQA
    """
    Plot utility by return percentile. The final percentile is the objective.
    """
    
    def __init__(self, *, fig, title, bins ): # NOQA
        
        self.bins   = bins
        self.line   = None
        self.line2  = None
        self.ax     = fig.add_subplot()            
        self.ax.set_title(title)
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
            self.line2 =  self.ax.plot( [x[-1]], [utility[-1]], "*", color=color_gains )[0]
            self.ax.plot( x, utility0, "-", label="payoff" )
            self.ax.plot( x, utility0*0., ":", color="black" )
            self.ax.legend()
            self.ax.set_xlim( 0.-0.1, 1.+0.1 )
        else:
            self.line.set_ydata( utility )
            self.line2.set_ydata( [utility[-1]] )

        min_ = min( np.min(utility), np.min(utility0) )
        max_ = max( np.max(utility), np.max(utility0) )
        dx   = max( max_ - min_, 0.0001 )
        min_ -= dx/20.
        max_ += dx/20.
        self.ax.set_ylim(min_,max_)            

# -------------------------------------------------------
# Show hedges by spot
# -------------------------------------------------------
        
class Plot_Activity_By_Spot_and_Time(object): # NOQA
    """
    Plot action or delta by spot return and time step.
    """
    
    def __init__(self, *, fig, title, bins, slices, which_inst, with_std ): # NOQA
        self.bins       = bins
        self.with_std   = with_std
        self.slices     = slices
        self.which_inst = which_inst
        self.lines      = None
        self.fills      = {}
        self.ax         = fig.add_subplot()
        self.ax.set_title(title)
        self.ax.set_xlabel("Spot return")
        self.timeixs    = None
        
    def update(self, *, P, actions, spot_all, spot_ret ):# NOQA
        assert len(actions.shape) == 3, "Actions are of wrong dimension"
        nTime    = actions.shape[1]
        slices   = min(self.slices,nTime)
        
        if self.timeixs is None:
            self.timeixs  = np.linspace(0,nTime-1,slices,endpoint=True,dtype=np.int32)
        assert len(self.timeixs) == slices, "Internal error: %ld != %ld" % (len(self.timeixs(),slices))
        
        actions  = actions[:,:,self.which_inst]
        first    = self.lines is None
        min_     = None
        max_     = None
        if first:
            self.lines = []
        else:
            assert len(self.lines) == slices, "Internal error: found %ld lines instead of %ld" % (len(self.lines), slices)

        for k in self.fills:
            self.fills[k].remove()
            
        for i,t in zip(range(len(self.timeixs)),self.timeixs):
            x     = spot_all[:,t] / spot_all[:,0] - 1.
            ixs   = np.argsort( x )
            x     = x[ixs]
            x     = mean_bins( x, bins=self.bins, weights=P )

            act_t,\
            std_t = mean_bins( actions[:,t][ixs], bins=self.bins, weights=P, return_std=True )
            r     = 2. * (1. - float(t+1) / float(actions.shape[1]))
            c1    = max(min(r-1.,1.0),0.)
            c2    = max(min(r,1.0),0.) 

            if first:
                self.lines.append( self.ax.plot( x, act_t, color=(1.,c1,c2), label=("%ld" % (t+1))  )[0])
            else:
                self.lines[i].set_ydata( act_t )

            if self.with_std:
                self.fills["step%03ld" % i]   = self.ax.fill_between( x, act_t-std_t, act_t+std_t, color=(1.,c1,c2), alpha=0.2 )
        
            min_ = np.min( act_t ) if min_ is None else min( min_, np.min(act_t) )
            max_ = np.max( act_t ) if max_ is None else max( max_, np.max(act_t) )

        if first:
            self.ax.legend()
        dx   = max( max_ - min_, 0.0001 )
        min_ -= dx/20.
        max_ += dx/20.
        self.ax.set_ylim(min_,max_)     
        
# -------------------------------------------------------
# Hedges by by time step
# -------------------------------------------------------

class Plot_Activity_By_Step(object): # NOQA
    """
    Plot action or delta by step.
    """
    
    def __init__(self, *, fig, activity_name, set_name, pcnt_lo, pcnt_hi, inst_names ): # NOQA
        
        self.pcnt_lo    = pcnt_lo
        self.pcnt_hi    = pcnt_hi
        self.inst_names = inst_names
        self.lines      = None
        self.fills      = None
        self.ax         = fig.add_subplot()
        self.ax.set_title("%s by time step\n(%s set)" % (activity_name,set_name))
        self.ax.set_xlabel("Step")
        
    def update(self, *, P, actions ):# NOQA

        nSamples = actions.shape[0] # NOQA
        nSteps   = actions.shape[1]
        nInst    = actions.shape[2]
        assert nInst == len(self.inst_names), "Internal error: %ld != %ld" % ( nInst, len(self.inst_names) )
        first    = self.lines is None
        
        # percentiles
        # -----------
        
        x = np.linspace(1,nSteps,nSteps,endpoint=True,dtype=np.int32)

        if self.lines is None:
            self.lines = []
            self.fills = []
            
        min_ = None
        max_ = None

        for iInst, color in zip( range(nInst), colors() ):
            action_i       = actions[:,:,iInst]
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
            
            min_ = np.min( percentiles_i ) if min_ is None else min( min_, np.min( percentiles_i ) )
            max_ = np.max( percentiles_i ) if max_ is None else max( max_, np.max( percentiles_i ) )

        if first:
            self.ax.legend()
        dx = max( max_ - min_, 0.0001)
        max_ += dx/20.
        min_ -= dx/20.
        self.ax.set_ylim( min_, max_ )

# -------------------------------------------------------
# Monitor plotter
# -------------------------------------------------------

class Plotter(object):
    """
    Object to print progress information during training.
    
    Contains plotting logic using 'dynaplot'.
    Add new plots here.
    """
    
    def __init__(self, plot_graphs : bool, config : Config):
        """
        Initialize Plooter

        Args:
            plot_graphs : bool
                Whether or not to plot matplotlib grapsh
            config : Config
                Configuration
        """
        self.fig = None
        self.plot_graphs      = plot_graphs
        self.epoch_refresh    = config("epoch_refresh", 10, Int>0, "Epoch fefresh frequency for visualizations" )        
        self.fig_row_size     = config.fig("row_size", 5, Int>0, "Plot size of a row")
        self.fig_col_size     = config.fig("col_size", 5, Int>0, "Plot size of a column")
        self.fig_col_nums     = config.fig("col_nums", 6, Int>0, "Number of columbs")
        self.err_dev          = config("err_dev", 1., Float>0., "How many standard errors to add to loss to assess best performance" )   
        self.lookback_window  = config("lookback_window", 200, Int>3, "Lookback window for determining y min/max in graphs.")
        self.show_epochs      = config("show_epochs", 100, Int>3,  "Maximum epochs displayed")
        self.bins             = config("bins", 100, Int>3, "How many x to plot")
        self.pcnt_lo          = config("confidence_pcnt_lo", 0.5, (Float > 0.) & (Float<=1.), "Lower percentile for confidence intervals")
        self.pcnt_hi          = config("confidence_pcnt_hi", 0.5, (Float > 0.) & (Float<=1.), "Upper percentile for confidence intervals")
        self.time_slices      = config("time_slices", 10, Int>0, "How many slice of spot action and delta to print")
        config.done()
    
    def __del__(self):#NOQA
        self.close()

    def close(self):
        """ Close the object """
        if not self.fig is None:
            self.fig.close()
            self.fig = None

    def __call__(self, *, world, val_world, last_cached_epoch, progress_data, training_info ):
        """ 
        Update our plots
        Create figures and subplots if not done so before
        """
        assert progress_data.epoch >= 0, "Do not call me before the first epoch"
        
        if self.plot_graphs:
            update_plots = progress_data.epoch == 0 or ((progress_data.epoch+1) % self.epoch_refresh == 0)
            if self.fig is None:
                """ Create figures """
                print("\r\33[2K" + (""*100))  # clear any previous text in this line
                update_plots = True
                # create figure
                self.fig                          = figure(row_size=self.fig_row_size, col_size=self.fig_col_size, col_nums=self.fig_col_nums, tight=True )
                
                # by epoch
                self.plot_loss_by_epoch           = Plot_Loss_By_Epoch(    fig=self.fig, title="Losses (recent)", epochs=training_info.epochs, err_dev=self.err_dev, lookback_window=self.lookback_window, show_epochs=self.show_epochs )
                self.plot_loss_by_epoch_all       = Plot_Loss_By_Epoch(    fig=self.fig, title="Losses (all)", epochs=training_info.epochs, err_dev=self.err_dev, lookback_window=self.lookback_window, show_epochs=training_info.epochs )
                self.plot_gains_utility_by_epoch  = Plot_Utility_By_Epoch( fig=self.fig, name="Model Gains", label="gains", err_dev=self.err_dev, epochs=training_info.epochs, lookback_window=self.lookback_window, show_epochs=self.show_epochs )
                self.plot_payoff_utility_by_epoch = Plot_Utility_By_Epoch( fig=self.fig, name="Original Payoff", label="payoff", err_dev=self.err_dev, epochs=training_info.epochs, lookback_window=self.lookback_window, show_epochs=self.show_epochs )
                self.plot_memory_by_epoch         = Plot_Memory_By_Epoch(  fig=self.fig, epochs=training_info.epochs )
    
                self.fig.next_row()
                
                # by performance - training
                # a key aspect of using utility based pricing is that when we solve \sup_a U( Z + a dH ), we will get a non-zere value u* = U( Z + a* dH ) at the optimal point a*.
                # This is the cash value of the position Z+a*dH, and it must be compared to the unhedged utility u0 = U(Z).
                
                self.plot_returns_by_spot_adj_ret      = Plot_Returns_By_Spot_Ret(      fig=self.fig, title = "Returns less Utility\n(training set)", bins=self.bins, with_std=False )
                self.plot_returns_by_spot_adj_ret_std  = Plot_Returns_By_Spot_Ret(      fig=self.fig, title = "Returns less Utility (with std)\n(training set)", bins=self.bins, with_std=True )
                self.plot_utility_by_cumpercentile     = Plot_Utility_By_CumPercentile( fig=self.fig, title = "Utility by cummulative percentile\n(training set)", bins=self.bins )
                
                # by performance - validation
                self.val_plot_returns_by_spot_adj_ret  = Plot_Returns_By_Spot_Ret(      fig=self.fig, title = "Returns less Utility\n(validation set)", bins=self.bins, with_std=False )
                self.val_plot_returns_by_spot_adj_ret_std = Plot_Returns_By_Spot_Ret(      fig=self.fig, title = "Returns less Utility (with std)\n(validation set)", bins=self.bins, with_std=True )
                self.val_plot_utility_by_cumpercentile = Plot_Utility_By_CumPercentile( fig=self.fig, title = "Utility by cummulative percentile\n(validation set)", bins=self.bins )
    
                self.fig.next_row()
    
                # activity by step
                self.plot_actions_by_step          = Plot_Activity_By_Step(    fig=self.fig, activity_name="Action", set_name="training", pcnt_lo=self.pcnt_lo, pcnt_hi=self.pcnt_hi, inst_names=world.inst_names )
                self.plot_deltas_by_step           = Plot_Activity_By_Step(    fig=self.fig, activity_name="Delta", set_name="training", pcnt_lo=self.pcnt_lo, pcnt_hi=self.pcnt_hi, inst_names=world.inst_names )
                # activity by spot
                self.plot_action0_by_step          = Plot_Activity_By_Spot_and_Time(    fig=self.fig, title="Spot action by time step\n(training set)", bins=self.bins, slices=self.time_slices, which_inst=0, with_std = False )
                self.plot_delta0_by_step           = Plot_Activity_By_Spot_and_Time(    fig=self.fig, title="Spot delta by time step\n(training set)", bins=self.bins, slices=self.time_slices, which_inst=0, with_std = False )
                # activity by spot, with std
                self.plot_action0_by_step_std      = Plot_Activity_By_Spot_and_Time(    fig=self.fig, title="Spot action by time step (with std)\n(training set)", bins=self.bins, slices=self.time_slices, which_inst=0, with_std = True )
                self.plot_delta0_by_step_std       = Plot_Activity_By_Spot_and_Time(    fig=self.fig, title="Spot delta by time step (with std)\n(training set)", bins=self.bins, slices=self.time_slices, which_inst=0, with_std = True )
    
                self.fig.render()
            
            if update_plots:
                # update live graphics
                # --------------------
                self.fig.suptitle("Learning to Trade, epoch %ld / %ld" % (progress_data.epoch+1,training_info.epochs), fontsize=20)
                
                # by epoch
                self.plot_loss_by_epoch.update( epoch=progress_data.epoch, losses=progress_data.losses, loss_errs=progress_data.losses_err, best_epoch=progress_data.best_epoch, best_loss=progress_data.best_loss )
                self.plot_loss_by_epoch_all.update( epoch=progress_data.epoch, losses=progress_data.losses, loss_errs=progress_data.losses_err, best_epoch=progress_data.best_epoch, best_loss=progress_data.best_loss )
                self.plot_gains_utility_by_epoch.update( epoch=progress_data.epoch, best_epoch=progress_data.best_epoch,  training_util=progress_data.utilities.training_util, training_util_err=progress_data.utilities.training_util_err, val_util=progress_data.utilities.val_util)
                self.plot_payoff_utility_by_epoch.update( epoch=progress_data.epoch, best_epoch=progress_data.best_epoch,  training_util=progress_data.utilities.training_util0, training_util_err=progress_data.utilities.training_util0_err, val_util=progress_data.utilities.val_util0)
                self.plot_memory_by_epoch.update( epoch=progress_data.epoch, process_info=progress_data.process )
                
                # by performance - training
                # Note that subtract the OCE utility from gains (the hedged portfolio) and payoff (the input).
                # Subtracting the OCE utility means that both are of equivalent utility.
                spot_ret             = world.details.spot_all[:,-1]      / world.details.spot_all[:,0] - 1.
                val_spot_ret         = val_world.details.spot_all[:,-1] / val_world.details.spot_all[:,0] - 1.
                
                adjusted_training_gains  = progress_data.training_result.gains  - mean(world.sample_weights, progress_data.training_result.utility)
                adjusted_training_payoff = progress_data.training_result.payoff - mean(world.sample_weights, progress_data.training_result.utility0)
                adjusted_training_hedge  = adjusted_training_gains - adjusted_training_payoff
                self.plot_returns_by_spot_adj_ret.update( P=world.sample_weights, gains=adjusted_training_gains, hedge=adjusted_training_hedge, payoff=adjusted_training_payoff, spot_ret=spot_ret )
                self.plot_returns_by_spot_adj_ret_std.update( P=world.sample_weights, gains=adjusted_training_gains, hedge=adjusted_training_hedge, payoff=adjusted_training_payoff, spot_ret=spot_ret )
                self.plot_utility_by_cumpercentile.update( P=world.sample_weights, utility=progress_data.training_result.utility, utility0=progress_data.training_result.utility0 )
                
                # by performance - validation
                adjusted_val_gains  = progress_data.val_result.gains  - mean(val_world.sample_weights, progress_data.val_result.utility)
                adjusted_val_payoff = progress_data.val_result.payoff - mean(val_world.sample_weights, progress_data.val_result.utility0)
                adjusted_val_hedge  = adjusted_val_gains - adjusted_val_payoff
                self.val_plot_returns_by_spot_adj_ret.update( P=val_world.sample_weights, gains=adjusted_val_gains, hedge=adjusted_val_hedge, payoff=adjusted_val_payoff, spot_ret=val_spot_ret)
                self.val_plot_returns_by_spot_adj_ret_std.update( P=val_world.sample_weights, gains=adjusted_val_gains, hedge=adjusted_val_hedge, payoff=adjusted_val_payoff, spot_ret=val_spot_ret)
                self.val_plot_utility_by_cumpercentile.update( P=val_world.sample_weights, utility=progress_data.val_result.utility, utility0=progress_data.val_result.utility0 )
                
                # activity by step
                assert len(progress_data.training_result.actions.shape) == 3, "Shape %s" % str(progress_data.training_result.actions.shape)
                self.plot_actions_by_step.update( P=world.sample_weights, actions=progress_data.training_result.actions )
                deltas = np.cumsum( progress_data.training_result.actions, axis=1 )
                self.plot_deltas_by_step.update( P=world.sample_weights, actions=deltas )
        
                # activity by time and spot
                self.plot_action0_by_step.update( P=world.sample_weights, actions=progress_data.training_result.actions, spot_all= world.details.spot_all, spot_ret=spot_ret )
                self.plot_action0_by_step_std.update( P=world.sample_weights, actions=progress_data.training_result.actions, spot_all= world.details.spot_all, spot_ret=spot_ret )
                self.plot_delta0_by_step.update( P=world.sample_weights, actions=deltas, spot_all= world.details.spot_all, spot_ret=spot_ret )
                self.plot_delta0_by_step_std.update( P=world.sample_weights, actions=deltas, spot_all= world.details.spot_all, spot_ret=spot_ret )
        
                self.fig.render()

        # Print
        # -----        
        training_loss_mean = progress_data.losses.training[-1]
        training_loss_err  = progress_data.losses_err.training[-1]
        val_loss_mean      = progress_data.losses.val[-1]
        val_loss_err       = progress_data.losses_err.val[-1]
        batch_loss         = progress_data.losses.batch[-1]
        
        # comment on timing:
        total_time_passed = sum( progress_data.times )
        time_per_epoch    = total_time_passed / float(progress_data.epoch+1)
        time_left         = time_per_epoch * max(0.,float(training_info.epochs-(progress_data.epoch+1)))
        when_done         = datetime.now() + timedelta( seconds = time_left )
        str_when_done     = ( ", estimated end time: %s" % fmt_datetime(when_done) ) if time_left > 0 else ""
        str_num_weights   = fmt_big_number( training_info.num_weights )
        
        str_sys   = "memory used: rss %gM, vms %gM" % ( progress_data.process.memory_rss[-1], progress_data.process.memory_vms[-1] )
        str_cache = "" if last_cached_epoch == -1 else (", last cached %ld" % (last_cached_epoch+1))
        str_intro = "Training %ld/%ld epochs; %s weights; %ld samples; %ld validation samples batch size %ld" % ( progress_data.epoch+1, training_info.epochs, str_num_weights, world.nSamples, val_world.nSamples, training_info.batch_size if not training_info.batch_size is None else 32)
        str_perf  = "initial loss %g (%g), training %g (%g), best %g (%g), batch %g, val %g (%g); best epoch %ld%s" % ( \
                                        progress_data.init_loss, progress_data.init_loss_err, \
                                        training_loss_mean, training_loss_err, \
                                        progress_data.best_loss, progress_data.best_loss_err, \
                                        batch_loss, \
                                        val_loss_mean, val_loss_err, \
                                        progress_data.best_epoch+1,\
                                        str_cache)
        str_time  = "time elapsed %s; time per epoch %s; estimated time remaining %s | current time: %s%s" % ( fmt_seconds(total_time_passed), fmt_seconds(time_per_epoch), fmt_seconds(time_left), fmt_now(), str_when_done )        
        print("\r\33[2K%s | %s | %s | %s                        " % ( str_intro, str_perf, str_sys, str_time ), end='')

    