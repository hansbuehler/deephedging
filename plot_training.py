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
from cdxbasics.util import uniqueHash
from cdxbasics.subdir import SubDir, uniqueFileName48, CacheMode
    
from .base import Logger, npCast, fmt_seconds, mean, err, tf, mean_bins, mean_cum_bins, perct_exp, Int, Float, fmt_big_number, fmt_list

_log = Logger(__file__)

colors = colors_tableau

# -------------------------------------------------------
# By epoch4b2a0d007d32efb21b4c813015b17235 
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
    Plot object for displaying utilities progress by epoch
    """
    
    def __init__(self, *, fig, name, err_dev, epochs, lookback_window, show_epochs ): # NOQA
        
        self.lookback_window  = lookback_window
        self.show_epochs      = show_epochs
        self.show_epochs      = min( self.show_epochs, epochs)
        self.err_dev          = err_dev
        self.lines            = None
        self.fills            = None
        self.ax               = fig.add_subplot()
        self.ax.set_title("%s Monetay Utility" % name)
        self.ax.set_xlabel("Epochs")
        
    def update(self, *, epoch, best_epoch, full_util, full_util_err, val_util ):# NOQA
    
        show_epoch0 = max( 0, epoch-self.show_epochs )
        show_epoch0 = min( best_epoch, show_epoch0 )
       
        x              = np.linspace(show_epoch0+1,epoch+1,epoch-show_epoch0+1,endpoint=True,dtype=np.int32 )
        best_full      = full_util[best_epoch]
        
        full_util      = np.array( full_util  )[show_epoch0:epoch+1]
        #full_util0     = np.array( full_util0 )[show_epoch0:epoch+1]
        full_util_err  = np.array( full_util_err  )[show_epoch0:epoch+1]
        #full_util0_err = np.array( full_util0_err )[show_epoch0:epoch+1]
        val_util       = np.array( val_util  )[show_epoch0:epoch+1]
        #val_util0      = np.array( val_util0 )[show_epoch0:epoch+1]        
        
        if self.lines is None:
            self.lines = pdct()
            self.lines.full_util  = self.ax.plot( x, full_util,  "-", label="gains, full", color="red" )[0]
            self.lines.val_util   = self.ax.plot( x, val_util,   ":", label="gains, val", color="red" )[0]
            self.lines.best       = self.ax.plot( [best_epoch+1], [best_full], "*", label="best", color="black" )[0]
            self.ax.legend()
            
        else:
            self.lines.full_util.set_ydata( full_util )
            self.lines.val_util.set_ydata( val_util )
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

        # y min/max            
        min_ = np.min(full_util[-self.lookback_window:])
        max_ = np.max(full_util[-self.lookback_window:])
        self.ax.set_ylim(min_-0.001,max_+0.001)
        self.ax.set_xlim(show_epoch0+1,show_epoch0+1+self.show_epochs)

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
    
    def __init__(self, *, fig, ret_name, set_name, bins ): # NOQA
        self.bins    = bins
        self.ax      = fig.add_subplot()            
        self.line    = None
        self.line_h  = None
        self.ax.set_title("%s by Spot Return\n(%s))" % (ret_name,set_name) )
        self.ax.set_xlabel("Spot return")
        
    def update(self, *, P, gains, hedge, payoff, spot_ret ):# NOQA

        ixs      = np.argsort( spot_ret )
        x        = spot_ret[ixs]
        gains    = gains[ixs]
        hedge    = hedge[ixs]
        payoff   = payoff[ixs]
        x        = mean_bins( x, bins=self.bins, weights=P )
        gains    = mean_bins( gains, bins=self.bins, weights=P )
        hedge    = mean_bins( hedge, bins=self.bins, weights=P )
        payoff   = mean_bins( payoff, bins=self.bins, weights=P )
        
        if self.line is None:
            self.line   =  self.ax.plot( x, gains, label="gains", color=color_gains )[0]
            self.ax.plot( x, payoff, ":", label="payoff", color=color_payoff )
            self.line_h =  self.ax.plot( x, -hedge, label="-hedge", color=color_hedge )[0]
            self.ax.plot( x, payoff*0., ":", color="black" )
            self.ax.legend()
        else:
            self.line.set_ydata( gains )
            self.line_h.set_ydata( -hedge )

        # below is pretty heuristic. If training works, below makes sense
        # as the gains process and the payoff are similar.
        # it produces less good visuals if training is bad ...
        min_ = min( np.min(gains), np.min(payoff) )
        max_ = max( np.max(gains), np.max(payoff) )
        if min_ >= max_-0.00001:
            min_ -= 0.0001
            max_ += 0.0001
        dx = (max_-min_)
        min_ -= 0.1*dx
        max_ += 0.1*dx
        self.ax.set_ylim(min_,max_)            

class Plot_Returns_By_Percentile(object): # NOQA
    """
    Plot object for showing hedging performance by percentile returns
    (not currently used)
    """
    
    def __init__(self, *, fig, set_name, bins ): # NOQA
        
        self.bins  = bins
        self.line1 = None
        self.line2 = None
        self.ax    = fig.add_subplot()            
        self.ax.set_title("Cash Returns by Percentile\n(%s)" % set_name )
        self.ax.set_xlabel("Percentile")
        
    def update(self, *, P, gains, payoff ):# NOQA

        gains   = np.sort( gains )
        payoff  = np.sort( payoff )
        gains   = mean_bins( gains, bins=self.bins, weights=P )
        payoff  = mean_bins( payoff, bins=self.bins, weights=P )
        
        x = np.linspace( 0., 1., self.bins, endpoint=True )
        
        if self.line1 is None:
            self.line1 =  self.ax.plot( x, gains, label="gains", color=color_gains )[0]
            self.line2 =  self.ax.plot( [x[-1]], [gains[-1]], "*", color=color_gains )[0]
            self.ax.plot( x, payoff, ":", label="payoff", color=color_payoff )
            self.ax.plot( x, payoff*0., ":", color="black" )
            self.ax.legend()
            self.ax.set_xlim( 0.-0.1, 1.+0.1 )
        else:
            self.line1.set_ydata( gains )
            self.line2.set_ydata( [gains[-1]] )

        min_ = min( np.min(gains), np.min(gains) )
        max_ = max( np.max(gains), np.max(gains) )
        if min_ >= max_-0.00001:
            min_ -= 0.0001
            max_ += 0.0001
        #self.ax.set_ylim(min_,max_)            

class Plot_Utility_By_CumPercentile(object): # NOQA
    """
    Plot utility by return percentile. The final percentile is the objective.
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

# -------------------------------------------------------
# Show hedges by spot
# -------------------------------------------------------
        
class Plot_Activity_By_Spot(object): # NOQA
    """
    Plot action or delta by spot return and time step.
    """
    
    def __init__(self, *, fig, bins, slices, which_inst, activity_name, inst_name, set_name ): # NOQA
        self.bins       = bins
        self.slices     = slices
        self.which_inst = which_inst
        self.inst_name  = inst_name
        self.lines      = None
        self.ax         = fig.add_subplot()
        self.ax.set_title("%s by spot for %s\n(%s set)" % (activity_name, inst_name, set_name))
        self.ax.set_xlabel("Spot return")
        
    def update(self, *, P, actions, spot_all, spot_ret ):# NOQA
        assert len(actions.shape) == 3, "Actions are of wrong dimension"
        nTime    = actions.shape[1]
        slices   = min(self.slices,nTime)
        timeixs  = np.linspace(0,nTime-1,slices,endpoint=True,dtype=np.int32)
        
        actions  = actions[:,:,self.which_inst]
        first    = self.lines is None
        min_     = None
        max_     = None
        if first:
            self.lines = []

        for i,t in zip(range(len(timeixs)),timeixs):
            x     = spot_all[:,t] / spot_all[:,0] - 1.
            ixs   = np.argsort( x )
            x     = x[ixs]
            x     = mean_bins( x, bins=self.bins, weights=P )

            act_t = mean_bins( actions[:,t][ixs], bins=self.bins, weights=P )
            r     = 2. * (1. - float(t+1) / float(actions.shape[1]))
            c1    = max(min(r-1.,1.0),0.)
            c2    = max(min(r,1.0),0.) 

            if first:
                self.lines.append( self.ax.plot( x, act_t, color=(1.,c1,c2), label=("%ld" % (t+1))  )[0])
            else:
                self.lines[i].set_ydata( act_t )

            min_ = np.min( act_t ) if min_ is None else min( min_, np.min(act_t) )
            max_ = np.max( act_t ) if max_ is None else max( max_, np.max(act_t) )

        if min_ >= max_-0.00001:
            min_ -= 0.0001
            max_ += 0.0001
        dx = (max_-min_)
        min_ -= 0.1*dx
        max_ += 0.1*dx
        self.ax.set_ylim(min_,max_)     
        self.ax.legend()
        
# -------------------------------------------------------
# Hedges by by time step
# -------------------------------------------------------

class Plot_Activity_By_Step(object): # NOQA
    """
    Plot action or delta by step.
    """
    
    def __init__(self, *, fig, activity_name, pcnt_lo, pcnt_hi, inst_names ): # NOQA
        
        self.pcnt_lo    = pcnt_lo
        self.pcnt_hi    = pcnt_hi
        self.inst_names = inst_names
        self.lines      = None
        self.fills      = None
        self.ax         = fig.add_subplot()
        self.ax.set_title("Activity by time step: %s" % activity_name)
        self.ax.set_xlabel("Step")
        
    def update(self, *, P, actions ):# NOQA

        nSamples = actions.shape[0]
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

        if first:
            self.ax.legend()

# -------------------------------------------------------
# Monitor plotter
# -------------------------------------------------------

class Plotter(object):
    """
    Contains plotting logic using 'dynaplot'.
    Add new plots here
    """
    def __init__(self):
        self.fig = None
    
    def __del__(self):
        self.close()
        
    def close(self):
        """ Close the object """
        if not self.fig is None:
            self.fig.close()
            self.fig = None

    def __call__(self, monitor, world, val_world, last_cached_epoch, cache_epoch_off ):
        """ 
        Update our plots
        Create figures and subplots if not done so before
        """
        assert monitor.epoch >= 0, "Do not call me before the first epoch"
        
        if self.fig is None:
            print("\r")
            # create figure
            self.fig                          = figure(row_size=monitor.fig_row_size, col_size=monitor.fig_col_size, col_nums=monitor.fig_col_nums, tight=True )
            
            # by epoch
            self.plot_loss_by_epoch           = Plot_Loss_By_Epoch(    fig=self.fig, title="Losses (recent)", epochs=monitor.epochs, err_dev=monitor.err_dev, lookback_window=monitor.lookback_window, show_epochs=monitor.show_epochs )
            self.plot_loss_by_epoch_all       = Plot_Loss_By_Epoch(    fig=self.fig, title="Losses (all)", epochs=monitor.epochs, err_dev=monitor.err_dev, lookback_window=monitor.epochs, show_epochs=monitor.epochs )
            self.plot_gains_utility_by_epoch  = Plot_Utility_By_Epoch( fig=self.fig, name="Gains", err_dev=monitor.err_dev, epochs=monitor.epochs, lookback_window=monitor.lookback_window, show_epochs=monitor.show_epochs )
            self.plot_payoff_utility_by_epoch = Plot_Utility_By_Epoch( fig=self.fig, name="Payoff", err_dev=monitor.err_dev, epochs=monitor.epochs, lookback_window=monitor.lookback_window, show_epochs=monitor.show_epochs )

            self.fig.next_row()
            
            # by performance - training
            # a key aspect of using utility based pricing is that when we solve \sup_a U( Z + a dH ), we will get a non-zere value u* = U( Z + a* dH ) at the optimal point a*.
            # This is the cash value of the position Z+a*dH, and it must be compared to the unhedged utility u0 = U(Z).
            
            self.plot_returns_by_spot_adj_ret      = Plot_Returns_By_Spot_Ret(      fig=self.fig, ret_name = "Returns less Utility", set_name="training set", bins=monitor.bins )
            self.plot_returns_by_spot_ret          = Plot_Returns_By_Spot_Ret(      fig=self.fig, ret_name = "Raw Returns", set_name="training set", bins=monitor.bins )
            self.plot_utility_by_cumpercentile     = Plot_Utility_By_CumPercentile( fig=self.fig, set_name = "training set", bins=monitor.bins )
            
            # by performance - validation
            self.val_plot_returns_by_spot_adj_ret  = Plot_Returns_By_Spot_Ret(      fig=self.fig, ret_name = "Returns less Utility", set_name="validation set", bins=monitor.bins )
            self.val_plot_returns_by_spot_ret      = Plot_Returns_By_Spot_Ret(      fig=self.fig, ret_name = "Raw Returns", set_name="validation set", bins=monitor.bins )
            self.val_plot_utility_by_cumpercentile = Plot_Utility_By_CumPercentile( fig=self.fig, set_name = "validation set", bins=monitor.bins )

            self.fig.next_row()

            # activity by step
            self.plot_actions_by_step          = Plot_Activity_By_Step(    fig=self.fig, activity_name="Actions\n(training set)", pcnt_lo=monitor.pcnt_lo, pcnt_hi=monitor.pcnt_hi, inst_names=world.inst_names )
            self.plot_deltas_by_step           = Plot_Activity_By_Step(    fig=self.fig, activity_name="Delta\n(training set)", pcnt_lo=monitor.pcnt_lo, pcnt_hi=monitor.pcnt_hi, inst_names=world.inst_names )
            # activity by spot
            self.plot_action0_by_step          = Plot_Activity_By_Spot(    fig=self.fig, bins=monitor.bins, slices=monitor.time_slices, which_inst=0, activity_name="Action", inst_name="spot", set_name="training")
            self.plot_delta0_by_step           = Plot_Activity_By_Spot(    fig=self.fig, bins=monitor.bins, slices=monitor.time_slices, which_inst=0, activity_name="Delta", inst_name="spot", set_name="training")

            self.fig.render()
        
        # update live graphics
        # --------------------
        self.fig.suptitle("Learning to Trade, epoch %ld / %ld" % (monitor.epoch+1,monitor.epochs), fontsize=20)
        
        # by epoch
        self.plot_loss_by_epoch.update( epoch=monitor.epoch, losses=monitor.losses, loss_errs=monitor.losses_err, best_epoch=monitor.best_epoch, best_loss=monitor.best_loss )
        self.plot_loss_by_epoch_all.update( epoch=monitor.epoch, losses=monitor.losses, loss_errs=monitor.losses_err, best_epoch=monitor.best_epoch, best_loss=monitor.best_loss )
        self.plot_gains_utility_by_epoch.update( epoch=monitor.epoch, best_epoch=monitor.best_epoch,  full_util=monitor.utilities.full_util, full_util_err=monitor.utilities.full_util_err, val_util=monitor.utilities.val_util)
        self.plot_payoff_utility_by_epoch.update( epoch=monitor.epoch, best_epoch=monitor.best_epoch,  full_util=monitor.utilities.full_util0, full_util_err=monitor.utilities.full_util0_err, val_util=monitor.utilities.val_util0)

        # by performance - training
        # Note that subtract the OCE utility from gains (the hedged portfolio) and payoff (the input).
        # Subtracting the OCE utility means that both are of equivalent utility.
        spot_ret             = world.details.spot_all[:,-1] / world.details.spot_all[:,0] - 1.
        val_spot_ret         = val_world.details.spot_all[:,-1] / val_world.details.spot_all[:,0] - 1.
        
        adjusted_full_gains  = monitor.full_result.gains  - mean(monitor.P, monitor.full_result.utility)
        adjusted_full_payoff = monitor.full_result.payoff - mean(monitor.P, monitor.full_result.utility0)
        adjusted_full_hedge  = adjusted_full_gains - adjusted_full_payoff
        self.plot_returns_by_spot_adj_ret.update( P=monitor.P, gains=adjusted_full_gains, hedge=adjusted_full_hedge, payoff=adjusted_full_payoff, spot_ret=spot_ret )
        self.plot_returns_by_spot_ret.update( P=monitor.P, gains=monitor.full_result.gains, hedge=monitor.full_result.gains - monitor.full_result.payoff, payoff=monitor.full_result.payoff, spot_ret=spot_ret )
        self.plot_utility_by_cumpercentile.update( P=monitor.P, utility=monitor.full_result.utility, utility0=monitor.full_result.utility0 )
        
        # by performance - validation
        adjusted_val_gains  = monitor.val_result.gains  - mean(monitor.val_P, monitor.val_result.utility)
        adjusted_val_payoff = monitor.val_result.payoff - mean(monitor.val_P, monitor.val_result.utility0)
        adjusted_val_hedge  = adjusted_val_gains - adjusted_val_payoff
        self.val_plot_returns_by_spot_adj_ret.update( P=monitor.val_P, gains=adjusted_val_gains, hedge=adjusted_val_hedge, payoff=adjusted_val_payoff, spot_ret=val_spot_ret)
        self.val_plot_returns_by_spot_ret.update( P=monitor.val_P, gains=monitor.val_result.gains, hedge=monitor.val_result.gains - monitor.val_result.payoff, payoff=monitor.val_result.payoff, spot_ret=val_spot_ret )
        self.val_plot_utility_by_cumpercentile.update( P=monitor.val_P, utility=monitor.val_result.utility, utility0=monitor.val_result.utility0 )
        
        # activity by step
        assert len(monitor.full_result.actions.shape) == 3, "Shape %s" % monitor.full_result.actions.shape
        self.plot_actions_by_step.update( P=monitor.P, actions=monitor.full_result.actions )
        self.plot_deltas_by_step.update( P=monitor.P, actions=monitor.full_result.deltas )

        # activity by time and spot
        self.plot_action0_by_step.update( P=monitor.P, actions=monitor.full_result.actions, spot_all= world.details.spot_all, spot_ret=spot_ret )
        self.plot_delta0_by_step.update( P=monitor.P, actions=monitor.full_result.deltas, spot_all= world.details.spot_all, spot_ret=spot_ret )

        self.fig.render()

        # Print
        # -----        
        full_loss_mean    = monitor.losses.full[-1]
        full_loss_err     = monitor.losses_err.full[-1]
        val_loss_mean     = monitor.losses.val[-1]
        val_loss_err      = monitor.losses_err.val[-1]
        batch_loss        = monitor.losses.batch[-1]
        
        # comment on timing:
        # in the event of being restored from caching, timing stats at 'cache_epoch_off'
        total_time_passed = time.time() - monitor.time0 
        time_per_epoch    = total_time_passed / float(monitor.epoch+1-cache_epoch_off) if monitor.epoch+1-cache_epoch_off > 0 else 0.
        time_left         = time_per_epoch * float(monitor.epochs-monitor.epoch)
        weights           = fmt_big_number( monitor.num_weights )
        
        str_cache = "" if last_cached_epoch == -1 else (" Last cached epoch %ld." % last_cached_epoch)
        str_intro = "Training %ld/%ld epochs; %s weights; %ld samples; %ld validation samples batch size %ld" % ( monitor.epoch+1, monitor.epochs, weights, world.nSamples, val_world.nSamples, monitor.batch_size)
        str_perf  = "initial loss %g (%g), full %g (%g), best %g (%g), batch %g, val %g (%g). Best epoch %ld.%s" % ( \
                                        monitor.init_loss, monitor.init_loss_err, \
                                        full_loss_mean, full_loss_err, \
                                        monitor.best_loss, monitor.best_loss_err, \
                                        batch_loss, \
                                        val_loss_mean, val_loss_err, \
                                        monitor.best_epoch,\
                                        str_cache)
        str_time  = "time elapsed %s; time per epoch %s; estimated time remaining %s" % ( fmt_seconds(total_time_passed), fmt_seconds(time_per_epoch), fmt_seconds(time_left) )        
        str_time  = str_time if monitor.time_out is None else str_time + ("; time out %ld" % fmt_seconds(monitor.time_out))
        print("\r%s | %s | %s                         " % ( str_intro, str_perf, str_time ), end='')

# -------------------------------------------------------
# Monitor
# -------------------------------------------------------

class NotebookMonitor(tf.keras.callbacks.Callback):
    """
    Monitors progress of our training and displays the result
    in a jupyter notebook with dynamic graphing.    
    
    "Base class" is trainer.NoMonitor
    """
    
    def __init__(self, gym, world, val_world, result0, epochs, batch_size, time_out, config_visual, config_caching ):# NOQA
        tf.keras.callbacks.Callback.__init__(self)
        
        """
        We store all data in 'self.data' in order to being able
        to load/save the current state of our training.
        This allows hot starting a run.
        """
                
        self.gym                   = gym  # do NOT store 'gym' in data as this cannot easily be pickled.
        self.data                  = pdct()
        self.data.time_refresh     = config_visual("time_refresh", 20, Int>0, "Time refresh interval for visualizations" )
        self.data.epoch_refresh    = config_visual("epoch_refresh", 10, Int>0, "Epoch fefresh frequency for visualizations" )        
        self.data.fig_row_size     = config_visual.fig("row_size", 5, Int>0, "Plot size of a row")
        self.data.fig_col_size     = config_visual.fig("col_size", 5, Int>0, "Plot size of a column")
        self.data.fig_col_nums     = config_visual.fig("col_nums", 6, Int>0, "Number of columbs")
        self.data.err_dev          = config_visual("err_dev", 1., Float>0., "How many standard errors to add to loss to assess best performance" )   
        self.data.lookback_window  = config_visual("lookback_window", 30, Int>3, "Lookback window for determining y min/max")
        self.data.show_epochs      = config_visual("show_epochs", 100, Int>3,  "Maximum epochs displayed")
        self.data.bins             = config_visual("bins", 200, Int>3, "How many x to plot")
        self.data.pcnt_lo          = config_visual("confidence_pcnt_lo", 0.5, (Float > 0.) & (Float<=1.), "Lower percentile for confidence intervals")
        self.data.pcnt_hi          = config_visual("confidence_pcnt_hi", 0.5, (Float > 0.) & (Float<=1.), "Upper percentile for confidence intervals")
        self.data.time_slices      = config_visual("time_slices", 10, Int>0, "How many slice of spot action and delta to print")
        config_visual.done()

        self.world                 = world
        self.val_world             = val_world
        self.data.result0          = result0
        self.data.num_weights      = gym.num_trainable_weights
        self.data.P                = world.sample_weights
        self.data.val_P            = val_world.sample_weights
        self.data.full_result      = None
        self.data.val_result       = None
        self.data.epochs           = epochs
        self.data.time_out         = time_out
        self.data.time0            = time.time()
        self.data.time_last        = -1
        self.data.batch_size       = batch_size if not batch_size is None else 32
        self.data.epoch            = -1
        self.data.why_stopped      = "Ran all %ld epochs" % epochs
        
        # track progress
        self.data.losses            = pdct()
        self.data.losses.init       = []
        self.data.losses.full       = []
        self.data.losses.batch      = []
        self.data.losses.val        = []
        self.data.losses_err        = pdct()
        self.data.losses_err.full   = []
        self.data.losses_err.val    = []

        self.data.init_loss         = mean( self.data.P, result0.loss )
        self.data.init_loss_err     = err( self.data.P, result0.loss )
        self.data.best_loss         = self.data.init_loss 
        self.data.best_loss_err     = self.data.init_loss_err
        self.data.best_weights      = self.gym.get_weights()
        self.data.best_epoch        = 0        
        
        self.data.utilities         = pdct()
        self.data.utilities.full_util      = []
        self.data.utilities.full_util0     = []
        self.data.utilities.full_util_err  = []
        self.data.utilities.full_util0_err = []
        self.data.utilities.val_util       = []
        self.data.utilities.val_util0      = []
        
        # plotting
        self.plotter              = Plotter()

        print("Network feature information:\n"\
              " Features used by the agent:        %s\n"\
              " Features available to the agent:   %s\n"\
              " Features used by the utility:      %s\n"\
              " Features available to the utility: %s" % \
      ( fmt_list(gym.agent_features_used), fmt_list(gym.available_features_per_step), fmt_list(gym.utility_features_used), fmt_list(gym.available_features_per_path)) )

        # restore training from cache
        # ---------------------------
        
        # caching, config
        self.cache_dir        = config_caching("directory", "~/dh_cache", str, "If specified, will use the directory to store a persistence file for the model")
        self.cache_mode       = config_caching("mode", CacheMode.ON, CacheMode.MODES, "Caching strategy: %s" % CacheMode.HELP)
        self.cache_freq       = config_caching("epoch_freq", 10, Int>0, "How often to cache results, in number of epochs")
        cache_file_name       = config_caching("debug_file_name", None, help="Allows overwriting the filename for debugging an explicit cached state")

        self.cache_mode       = CacheMode( self.cache_mode )
        self.cache_dir        = SubDir(self.cache_dir, "!")
        optimizer_id          = uniqueHash( tf.keras.optimizers.serialize( gym.optimizer ) )
        self.cache_file       = uniqueFileName48( gym.unique_id, optimizer_id, world.unique_id, val_world.unique_id ) if cache_file_name is None else cache_file_name
        self.full_cache_file  = self.cache_dir.fullKeyName( self.cache_file )
        self.cache_epoch_off  = 0    # how many epochs have been restored from cache, if any
        self.cache_last_epoch = -1   # last restored or written epoch
        config_caching.done()

        if not self.cache_mode.is_off:
            print("Caching enabled @ '%s'" %  self.full_cache_file)
            if self.cache_mode.delete:
                self.cache_dir.delete( self.cache_file )    
            elif self.cache_mode.read:
                # restore cache                
                cache = self.cache_dir.read( self.cache_file )
                if not cache is None:
                    # load everything except the gym 
                    monitor_cache = cache['monitor']
                    for k in self.data:
                        assert k in monitor_cache, "Consistency error: self.data contains key '%s' (type %s), which is not contained in the restored cache" % (k, type(self.data[k]))
                    # restore gym
                    if not self.gym.restore_from_cache( cache['gym'], world ):
                        print("\rCache consistency error: could not write weights from cache to current model. This is most likely because the model architecture changed.\n"\
                              "Use config.train.caching.mode = 'renew' to rebuild the cache if this is the case. Use config.train.caching.mode = 'off' to turn caching off.\n")
                    else:
                        for k in monitor_cache:
                            assert k in self.data, "Could not find '%s' in self.data?" % k
                            self.data[k] = monitor_cache[k]
                        _log.verify( self.data.epoch >= 0, "Error: object restored from cache had epoch set to %ld", self.data.epoch )
                        # tell world that we were restored
                        self.cache_last_epoch = self.data.epoch
                        self.cache_epoch_off  = self.data.epoch+1
                        self.data.epochs      = epochs
                        print("Cache successfully loaded. Current epoch: %ld" % self.data.epoch )

        # initialize timing
        self.data.time0            = time.time()
        self.data.time_last        = self.cache_last_epoch    
        if self.cache_epoch_off >= epochs:
            print( "Nothing to do: cached model loaded from %s represents a trained model up to %ld epochs (you have asked to train for %ld epochs). "\
                   "Raise number of epochs or turn off caching to re-start training.\n\nPlotting results for the trained model.\n" % \
                   ( self.full_cache_file, self.cache_epoch_off, epochs ) )
        else:
            remaining = "" if self.data.epoch == -1 else "remaining "
            weights   = fmt_big_number( self.data.num_weights )
            
            print("\rDeep Hedging Engine: warming up to train %s weights using %ld %sepochs over %ld samples and %ld validation samples ...         " % (weights,epochs-(self.data.epoch+1), remaining, world.nSamples, self.val_world.nSamples), end='')

    @property
    def is_done(self):
        return self.data.epoch+1 >= self.data.epochs
    
    def on_epoch_begin( self, epoch, logs = None ):# NOQA
        if self.data.epoch == -1:
            weights   = fmt_big_number( self.data.num_weights )
            print("\r\33[2KDeep Hedging Engine: first of %ld epochs for training %s weights over %ld samples and %ld validation samples started. Compiling graph ...       " % (self.data.epochs, weights, self.world.nSamples, self.val_world.nSamples), end='')
        epoch                 = epoch + self.cache_epoch_off # cached warm start
            
    def on_epoch_end( self, epoch, logs = None ):
        """ Called when an epoch ends """
        if self.data.epoch == -1:
            empty = " "*200
            print("\r\33[2K "+empty+"\r", end='')
        
        epoch                 = epoch + self.cache_epoch_off # cached warm start
        assert len(self.data.losses.batch) == epoch, "Internal error: expected %ld losses. Found %s. Cache Epoch is %ld" % ( epoch,len(self.data.losses.batch),self.cache_epoch_off)
        self.data.full_result = npCast( self.gym(self.world.tf_data) )
        self.data.val_result  = npCast( self.gym(self.val_world.tf_data) )
        self.data.epoch       = epoch

        # losses
        # Note that we apply world.sample_weights to all calculations
        # so we are in sync with keras.fit()
        self.data.losses.batch.append(   float( logs['loss_default_loss'] ) ) # we read the metric instead of 'loss' as this appears to be weighted properly
        self.data.losses.full.append(    mean(self.data.P, self.data.full_result.loss) )
        self.data.losses.val.append(     mean(self.data.val_P, self.data.val_result.loss) )
        self.data.losses.init.append(    self.data.init_loss )

        self.data.losses_err.full.append( err(self.data.P,self.data.full_result.loss) )
        self.data.losses_err.val.append(  err(self.data.val_P,self.data.val_result.loss) )
        
        if self.data.losses.full[-1] < self.data.best_loss:
            self.data.best_loss         = self.data.losses.full[-1]
            self.data.best_loss_err     = self.data.best_loss_err
            self.data.best_weights      = self.gym.get_weights()
            self.data.best_epoch        = epoch
            
        # utilities
        self.data.utilities.full_util.append(     mean(self.data.P, self.data.full_result.utility ) )
        self.data.utilities.full_util0.append(    mean(self.data.P, self.data.full_result.utility0) )
        self.data.utilities.full_util_err.append( err( self.data.P, self.data.full_result.utility ) )
        self.data.utilities.full_util0_err.append(err( self.data.P, self.data.full_result.utility0) )
        self.data.utilities.val_util.append(      mean(self.data.val_P, self.data.val_result.utility ) )
        self.data.utilities.val_util0.append(     mean(self.data.val_P, self.data.val_result.utility0 ) )
        
        # cache or not
        # ------------
        
        if epoch % self.cache_freq == 0 and self.cache_mode.write and epoch > self.cache_last_epoch:
            assert self.data.epoch >= 0, "Internal error: epoch %ld should not happen" % self.data.epoch
            cache = { 'gym':     self.gym.create_cache(),
                      'monitor': self.data
                    }            
            self.cache_dir.write( self.cache_file, cache )
            self.cache_last_epoch = self.data.epoch
        
        # print or not
        # ------------
        
        total_time_passed = time.time() - self.data.time0 
        if not self.data.time_out is None and total_time_passed > self.data.time_out:
            print("\r\33[2K**** Training timed out after a run time of %s" % fmt_seconds(total_time_passed))
            self.model.stop_training = True
            self.data.why_stopped    = "Timeout after %s" % fmt_seconds(total_time_passed)
        elif epoch != 0 and \
            epoch % self.data.epoch_refresh != 0 and \
            ( time.time() - self.data.time_last ) < self.data.time_refresh:
            return
        
        self.plotter(self.data, self.world, self.val_world , last_cached_epoch = self.cache_last_epoch, cache_epoch_off = self.cache_epoch_off )

    def finalize( self ):
        """ Plot final result and copy best weights to model """
        
        # tell user what happened
        if self.data.why_stopped == "Aborted":
            print("\r                                      \r*** Aborted ... ", end='')
        else:
            empty = " "*200
            print("\r\33[2K "+empty+"\r", end='')

        # cache current state /before/ we reset gym to its best weights
        # this way we can continue to train from where we left it
        cached_msg = ""
        if self.data.epoch >= 0 and self.data.epoch != self.cache_last_epoch and self.cache_mode.write:
            cache = { 'gym':     self.gym.create_cache(),
                      'monitor': self.data
                    }            
            self.cache_dir.write( self.cache_file, cache )
            self.cache_last_epoch = self.data.epoch
            cached_msg = " State of training until epoch %ld cached into %s\n" % (self.cache_last_epoch, self.full_cache_file)

        # restore best weights
        self.gym.set_weights( self.data.best_weights )
        self.data.full_result = npCast( self.gym(self.world.tf_data) )
        self.data.val_result  = npCast( self.gym(self.val_world.tf_data) )

        # upgrade plot
        self.plotter(self.data, self.world, self.val_world , last_cached_epoch = self.cache_last_epoch, cache_epoch_off = self.cache_epoch_off)
        print("\n Status: %s.\n Weights set to best epoch: %ld\n%s" % (self.data.why_stopped, self.data.best_epoch,cached_msg) )
        self.plotter.close()
        