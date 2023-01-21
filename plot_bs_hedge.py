
from deephedging.base import npCast, perct_exp, mean
from deephedging.trainer import train_utillity
from cdxbasics.dynaplot import figure
from cdxbasics.config import Config
import numpy as np
import math as math
import tqdm as twdm
from scipy.stats import norm

def plot_blackscholes( world, gym, config, strike : float = 1., iscall : bool = True ):
    """
    Simple utility function to plot BS hedge performance.
    This function assumes that DH was used to hedge a vanilla European option
    
    Parameters
    ----------
        world : world
        gym : gym
        iscall : bool
        strike : float
    """
    # get results
    r       = gym(world.tf_data)
    spot    = world.details.spot_all[:,:-1]
    hedges  = world.data.market.hedges[:,:,0]
    costs   = world.data.market.cost[:,:,0]
    actions = npCast( r.actions )[:,:,0]   # only one asset
    deltas  = np.cumsum( actions, axis=1 )
    dhpnl   = npCast( r.pnl )
    payoff  = npCast( r.payoff )

    utility  = mean( world.sample_weights, npCast( r.utility ) )
    utility0 = mean( world.sample_weights, npCast( r.utility0 ) )
    dprice   = utility - utility0
    
    assert actions.shape == spot.shape, "Error: expected 'spots' and 'actions' to have same dimension. Found %s and %s" % (spot.shape,actions.shape)
    assert hedges.shape == spot.shape, "Error: expected 'spots' and 'hedges' to have same dimension. Found %s and %s" % (spot.shape,hedges.shape)
    
    # load data for BS computation
    dt       = config.world.get_raw("dt", 1./50.)
    vol      = config.world.get_raw("rvol", 0.2)
    #drift    = config.world.get_recorded("drift")  # WARNING. This is the real-life drift *NOT* the risk-neutral drift.
    
    # prep spot plotting
    bins          = 20
    nSpots        = spot.shape[0]
    spotT         = world.details.spot_all[:,-1]
    srt_spotT     = np.sort( spotT )
    lohi          = perct_exp( srt_spotT, 0.01, .99 )
    binBnd        = np.linspace( lohi[0], lohi[1], bins+1, endpoint=True )
    binMid        = 0.5 *( binBnd[1:] + binBnd[:-1] )
    
    def spot_histogram( spots_t ):
        res = np.zeros((bins))
        n   = 0
        for i in range(bins):
            _ = spots_t[ spots_t >= binBnd[i] ]
            _ = _[ _<binBnd[i+1] ]
            c      = np.sum(_)
            res[i] = float(c)
            n      += c
        res = res/float(n)
        return res
    
    delta_bins = 100
    bin_ixs    = np.linspace(0,nSpots,delta_bins+1,endpoint=True,dtype=np.int32)
    
    # start plotting
    fig_any = figure()
    
    plt_termpayoff = fig_any.add_subplot()
    plt_termpayoff.set_title("Effective Terminal Payoffs")
    plt_terminal = fig_any.add_subplot()
    plt_terminal.set_title("Terminal Hedged Results")
    plt_utility = fig_any.add_subplot()
    plt_utility.set_title("Utility %s\n(higher is better)" % gym.utility.display_name)
    plt_spots = fig_any.add_subplot()
    plt_spots.set_title("Spot distribution in t")    
    plt_hedges = fig_any.add_subplot()
    plt_hedges.set_title("Hedge Returns")

    fig_path  = figure(col_nums =6)
    
    time_steps = deltas.shape[1]
    
    last_delta = 0.
    last_bsdelta = 0.
    pnl = 0.
    cost = 0.
    bspnl = 0.
    bscost = 0.
    
    print("Running strategies ...", end='')
    for j in range(time_steps):
    
        # sort by spot at j, and compute BS refernece
        spot_t    = spot[:,j]
        delta_t   = deltas[:,j]
        hedges_t  = hedges[:,j]
        cost_t    = costs[:,j]
        t         = float(j) * dt
        res_t     = float(time_steps-j) * dt 
        
        # BS
        # note that the 'drift' in the simulator is the statistical drift, not the risk-neutral drift.
        d1            = ( np.log(spot_t/strike) +  0.5 * vol * vol  * res_t ) / math.sqrt( res_t * vol * vol )
        d2            = d1 - vol * math.sqrt( res_t )
        N1            = norm.cdf(d1)
        N2            = norm.cdf(d2)
        bsprice_t     = spot_t * N1 - strike * N2
        bsprice_t     = bsprice_t if iscall else bsprice_t + strike - spot_t    # C=P+S-K
        bsdelta_t     = N1
        bsdelta_t     = bsdelta_t if iscall else bsdelta_t - 1.
        
        # action
        act_t         = delta_t - last_delta 
        bsact_t       = bsdelta_t - last_bsdelta
        pnl           = act_t * hedges_t + pnl
        bspnl         = bsact_t * hedges_t + bspnl
        cost          = np.abs(act_t) * cost_t
        bscost        = np.abs(bsact_t) * cost_t
        last_delta    = delta_t
        last_bsdelta  = bsdelta_t
        
        # approximat price
        sim_price_t   = pnl + delta_t * ( spotT - spot_t )
        
        # sort data
        ixs           = np.argsort(spot_t)
        srt_spot_t    = spot_t[ixs]
        srt_delta_t   = delta_t[ixs]
        srt_bsdlt_t   = bsdelta_t[ixs]
        srt_hedges_t  = hedges_t[ixs]
        src_simpr_t   = sim_price_t[ixs]
        src_bsprc_t   = bsprice_t[ixs]
        
        # compute averages over sample data
        bin_spot_t  = 0.5 * (  srt_spot_t[bin_ixs[:-1]]+srt_spot_t[np.minimum(nSpots-1,bin_ixs[1:])] )
        bin_spot_t  = np.array([ np.mean(srt_spot_t[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
        bin_delta_t = np.array([ np.mean(srt_delta_t[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
        bin_bsdlt_t = np.array([ np.mean(srt_bsdlt_t[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
        bin_simpr_t = np.array([ np.mean(src_simpr_t[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
        bin_bsprc_t = np.array([ np.mean(src_bsprc_t[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
                               
        # plot spot
        plt_spots.plot( binMid, spot_histogram( np.sort(spot_t)), color=(float(j+1)/float(time_steps), 0.5, 0.5), label="%g days" % (t*255) )
        # plot hedges
        plt_hedges.plot( srt_spot_t, srt_hedges_t, color=(0.7, float(j+1)/float(time_steps), 0.7) if j<time_steps-1 else (0.8,1,0.8))
    
        # plot prices
        plt = fig_path.add_subplot()
        plt.plot( bin_spot_t, bin_simpr_t,  "-" if j>0 else "o", label="Model approximation", color=(0.,0,1))
        plt.plot( bin_spot_t, bin_bsprc_t - dprice,  "-" if j>0 else "o", label="Black Scholes", color=(0.8,0.,0.))
        plt.set_title("-Payoff %g days" % (t*255))
        if j == 1:
            plt.legend()

        # plot deltas
        plt = fig_path.add_subplot()
        plt.plot( srt_spot_t, srt_delta_t, "-" if j>0 else "o", label="model", color=(0.,0.,1.), alpha=0.3)
        plt.plot( bin_spot_t, bin_delta_t,  "-" if j>0 else "o", label="model (smoothed)", color=(0.,0.,0.5))
        plt.plot( srt_spot_t, srt_bsdlt_t,  "-" if j>0 else "o", label="black scholes", color=(0.8,0.,0.))
        plt.set_title("Delta %g days" % (t*255))
        if j == 1:
            plt.legend()
            
    print("done")
            
    # compute terminal utility for BS
    tconfig = Config()
    tconfig.train.optimizer.name = "adam"
    tconfig.train.batch_size = world.nSamples # full sample size
    tconfig.train.epochs = 100
    tconfig.train.tf_verbose = 0

    print("Running tensorflow to compute utility of BS strategy ...", end='')
    utilityBS = train_utillity( gym.utility, world, payoff = payoff, pnl = bspnl, cost = bscost, config=tconfig )
    utilityBS = utilityBS[0]
    print("done; result %g", utilityBS)

    # bin pnl
    ixs        = np.argsort( spotT )            
    srt_spotT  = spotT[ixs]
    srt_gain   = (pnl + payoff)[ixs]
    srt_bsgain = (bspnl + payoff)[ixs]
    srt_dhgain = (dhpnl + payoff)[ixs]
    srt_payoff = (payoff)[ixs]
    srt_eff    = pnl[ixs]
    srt_dheff  = dhpnl[ixs]
    srt_bseff  = bspnl[ixs]
    
    bin_spotT  = np.array([ np.mean(srt_spotT[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
    bin_gain   = np.array([ np.mean(srt_gain[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
    bin_bsgain = np.array([ np.mean(srt_bsgain[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
    bin_dhgain = np.array([ np.mean(srt_dhgain[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
    
    min_ = min( np.min(bin_gain), np.min(bin_bsgain), np.min(bin_dhgain) )
    max_ = max( np.max(bin_gain), np.max(bin_bsgain), np.max(bin_dhgain) )
    dx   = max_ - min_
    min_ -= dx*0.25
    max_ += dx*0.25
    
    plt_terminal.plot( bin_spotT, bin_gain, "*-", color="orange", label="hedged pnl")
    plt_terminal.plot( bin_spotT, bin_bsgain, "-",   color="green",  label="bs hedged pnl")
    plt_terminal.plot( bin_spotT, bin_dhgain, ":", color="black", label="hedged pnl from DH")
    plt_terminal.legend()
    plt_terminal.set_ylim(min_,max_)

    bin_payoff = np.array([ np.mean(srt_payoff[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
    bin_eff    = np.array([ np.mean(srt_eff[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
    bin_dheff  = np.array([ np.mean(srt_dheff[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])
    bin_bseff  = np.array([ np.mean(srt_bseff[ bin_ixs[i]:bin_ixs[i+1]]) for i in range(delta_bins) ])

    min_ = min( np.min(bin_payoff), np.min(bin_eff), np.min(bin_dheff), np.min(bin_bseff) )
    max_ = max( np.max(bin_payoff), np.max(bin_eff), np.max(bin_dheff), np.max(bin_bseff) )
    dx   = max_ - min_
    min_ -= dx*0.25
    max_ += dx*0.25
    
    plt_termpayoff.plot( bin_spotT, bin_payoff, "-", color="blue", label="payoff")
    plt_termpayoff.plot( bin_spotT, -bin_eff   -dprice,  "*-", color="orange", label="-model hedged pnl")
    plt_termpayoff.plot( bin_spotT, -bin_bseff -dprice,  "-",  color="green", label="-bs hedged pnl")
#deep hedging    plt_termpayoff.plot( bin_spotT, -bin_dheff -dprice,  ":",  color="black", label="-hedged pnl from DH")
    plt_termpayoff.legend()
    plt_termpayoff.set_ylim(min_,max_)
        
    plt_utility.bar( ['Unhedged','BS','DH'], [ utility0, utilityBS, utility ] )
                
    print("Rendering now. Stand by.")
    
    fig_any.render()
    fig_any.close()
    
    fig_path.render()
    fig_path.close()
        

    