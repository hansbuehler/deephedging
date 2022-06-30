# -*- coding: utf-8 -*-
"""
Deep Hedging Trainer
--------------------
Training loop with visualization for 

June 30, 2022
@author: hansbuehler
"""

import numpy as np
import math as math
from packages.cdxbasics.cdxbasics.prettydict import PrettyDict as pdct
from packages.cdxbasics.cdxbasics.dynaplot import colors_tableau
from .base import Logger, mean_bins, mean_cum_bins, perct_exp

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
# By terminal outcome
# -------------------------------------------------------

