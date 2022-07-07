# Deep Hedging
## Reinforcement Learning for Hedging Derviatives under Market Frictions
### Beta version. Please report any issues.

This archive contains a sample implementation of of the Deep Hedging (http://deep-hedging.com) framework.
The notebook directory has a number of examples on how to use it. The framework relies on the pip package cdxbasics.

The Deep Hedging problem for a horizon $T$ is given as
<P>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$ \max_{a}: U[ \
        Z_T + \sum_{t=0}^{T-1} a(f_t) \cdot DH_t - | a(f_t)\gamma_t|
     \ \right]  $$
<p>
where  $DH_t:=H_T - H_t$ denotes the vector of returns of the hedging instruments to $T$. Cost are proportional.
The policy $a$ is a neural network which is fed both pre-computed and live features $f_t$ at each time step.
<p>
In order to run the Deep Hedging, we require:
<ol>
    <li><b>Market data</b>: this is referred to as a <tt>world</tt>. Among other members, world has a <tt>td_data</tt> member which
        represents the feature sets across training samples, and <tt>tf_sample_weights</tt> which is the probability distribution
        across samples. The code provides a simplistic default implementation, but for any real application it is recommend to rely
        on fully machine learned market simulators such as https://arxiv.org/abs/2112.06823.
    </li>
    <li><b>Gym</b>: essentially the <i>model</i>. This is more complex here than in standard ML, as we will construct our
        own Monte Carlo loop arund the actual, underlying networks.<br>
        Given a <tt>world</tt> we can compute the loss given the prevailing action network as <tt>gym(world.tf_data)</tt>.
    </li>
    <li><b>Train</b>: some cosmetics around <tt>keras.fit()</tt> with some nice live visualization using matplotlib if you 
        are in jupyter.
    </li>
</ol>

To provide your own world with real or simulator data, see <tt>world.py</tt>.
To give an indication of what is required, here are <tt>world.tf_data</tt> entries which are used by the <tt>gym</tt>:
<ul>
<li>
<tt>data['martket']['payoff']</tt> (:,M)<br> 
The payoff $Z_T$ at maturity. Since this is at or part the expiry of the product, this can be computed off the path until $T$.
<br>&nbsp;
</li>
<li>
<tt>data['martket']['payoff']</tt> (:,M,N)<br>
Returns of the hedges, i.e. the vector $DH_t:=H_T - H_t$. That means $H_t$ is the model price at time $t$, and $H_T$ is the price at time $T$. 
In most applications $T$ is chosen such that $H_T$
is the actual payoff.<br>
    For example, if $S_t$ is spot, ${\sigma_t}$ is an implied volatility,  $\tau$ is the time-to-maturity, and
    $k$ a relative strike, then $H_t = \mathrm{BSCall}( S_t, \sigma_t; \tau, kS_t )$ and $H_T = ( S_{t+\tau} / S_t - k )^+$.
<br>&nbsp;
</li>
<li>
<tt>data['martket']['payoff']</tt> (:,M,N)<br>
Cost $\gamma_t$ of trading the hedges in $t$ for proportional cost $c_t(a) = \gamma_t\cdot |a|$. 
More advanced implementations allow to pass the cost function itself as a tensorflow model.<br>
    In the simple setting an example for the cost of trading a vanilla call could be $\gamma_t = \gamma^\mathrm{Delta} \mathrm{BSDelta}(t,\cdots) 
    + \gamma^\mathrm{Vega}  \mathrm{BSVega}(t,\cdots)$.
<br>&nbsp;
</li><li>
<tt>data['martket']['unbd_a'], data['martket']['lnbd_a']</tt> (:,M,N)<br>
Min/max allowed action per time step: $a^\min_t \leq a \leq a^\max_t$, componenwise.
<br>&nbsp;

</li><li>
<tt>data['features']['per_step']</tt> (:,M,N)<br>
Featues for feeding the action network per time step such as current spot, current implied volatilities, time of the day, etx
<br>&nbsp;
</li><li>
<tt>data['features']['per_sample']</tt> (:,M)<br>
Featues for feeding the action network which are constant along the path such as features of the payoff, risk aversion, 
</ul>

The code examples provided are fairly general and allow for a wide range of applications. 
An example world generator for simplistic model dynamics is provided, but in practise it is recommend to rely
on fully machine learned market simulators such as https://arxiv.org/abs/2112.06823

## Installation

<ul>
    <li>Pip install <tt>cdxbasics</tt> version 0.1.3 or higher
    <li>Install TensorFlow 2.7 or higher
    <li>Install tensorflow_probability 0.15 or higher
    <li>Download this git directory in your Python path such that <tt>import deephedging.world</tt> works.
    <li>Open notebooks/trainer.ipynb</tt> and run it. If it 
</ul>

## Industrial Machine Learning Code Philosophy

We attempted to provide a base for industrial code development.
<ul>
    <li>
        <b>Notebook-free</b>: all code can, and is meant to run 
        outside a jupyter notebook. Notebooks are good for playing around but should not feature in any production environment.
        Notebooks are used for demonstration only.
        <br>&nbsp;
    </li>
    <li>    
        <b>Defensive programming</b>: validate as many inputs to functions as reasonable with clear, actionable, context-dependent 
        error messages. We use the <tt>cdxbasics.logger</tt> framework with a similar usage paradigm as C++ ASSSET/VERIFY.
        <br>&nbsp;
    </li>
    <li>
    <b>Robust configs</b>: all configurations of all objects are driven by dictionaries.
    However, the use of simple dictionaries leads to a number of inefficiencies which can slow down development.
    We therefore use <tt>cdxbasics.config.Config</tt> which provides:
    <ul>
        <li><b>Catch Spelling Errors</b>: ensures that any config parameter is understood by the receving code. 
            That means if the code expects <tt>config['nSamples']</tt> but we passed <tt>config['samples']</tt>
            an error is thrown.
        </li>
        <li><b>Self-documentation</b>: once parsed by receving code, the config is self-documenting and is able
            to print out any values used, including those which were not set by the users when calling the receiving code.
            </li>
        <li><b>Object notation</b>: it is a matter of tastes, but we prefer using <tt>config.nSamples = 2</tt> instead
            of the standard dictionary notation.
        </li>
        &nbsp;
    </ul>
    </li>
    <li>
        <b>Config-driven built</b>: 
        this avoids difference between training and inference code. In both cases, the respective model hierarchy is built
        from the configs up. No need to know in advance which sub-models where used within the overall model hierarchy.          
        <br>&nbsp;
    </li>
</ul>

## Key Objects and Functions

<ul>
    <li><b>world.SimpleWorld_Spot_ATM</b> class<br>
         Simple World with one asset and one floating ATM option.
    The asset has stochastic volatility, and a mean-reverting drift.
    The implied volatility of the asset is not the realized volatility, allowing to re-create some results from https://arxiv.org/abs/2103.11948
    
    Set the <tt>black_scholes</tt> boolean config flag to <tt>True</tt> to turn the world into a simple black & scholes world, with no traded option.
    Otherwise, use <tt>no_stoch_vol</tt> to turn off stochastic vol, and <tt>no_stoch_drift</tt> to turn off the stochastic mean reverting drift of the asset.
    If both are True, then the market is Black & Scholes, but the option can still be traded for hedging.
    
        <br>
    See <tt>notebooks/simpleWorld_Spot_ATM.ipynb</tt>
      <br>&nbsp;  
    </li>
    <li><b>gym.VanillaDeepHedgingGym</b> class<br>
        Main Deep Hedging training gym (the Monte Carlo). It will create internally the agent network and the monetary utility $U$.
        <br>
        To run the models for all samples of a given <tt>world</tt> use <tt>r = gym(world.tf_data)</tt>.<br>
        The returned dictionary contains the following members
        <ol>
                 <li><tt>utiliy:  </tt> (,) primary objective to maximize
            </li><li><tt>utiliy0: </tt> (,) objective without hedging
            </li><li><tt>loss:    </tt> (,) -utility-utility0
            </li><li><tt>payoff:  </tt> (,) terminal payoff 
            </li><li><tt>pnl:     </tt> (,) mid-price pnl of trading (e.g. ex cost)
            </li><li><tt>cost:    </tt> (,) cost of trading
            </li><li><tt>gains:   </tt> (,) total gains: payoff + pnl - cost 
            </li><li><tt>actions: </tt> (,M,N) actions, per step, per path
            </li><li><tt>deltas:  </tt> (,M,N) deltas, per step, per path
            </li>
        </ol>
        See <tt>notebooks/trainer.ipynb</tt>.
    </li>
        <br>
    The core engine in <tt>call()</tt> is not only about 200 lines of code. It is recommended to read it before using the framework.
      <br>&nbsp;  
    </li>
    <li><b>trainer.train</b> function<br>
        Main Deep Hedging training engine (stochastic gradient descent). <br>
        Trains the model using keras. Any optimizer supported by Keras might be used. When run in a Jupyer notebook the model will 
        dynamically plot progress in a number if graphs which will aid understanding on how training is progressing.<br>
    When training outside jupyer, set <tt>config.visual.monitor_type = "none"</tt> (or write your own).
    <br>
        See <tt>notebooks/trainer.ipynb</tt>.
    <br>
    <br>
    The <tt>train()</tt> function is barely 50 lines. It is recommended to read it before using the framework.
    </li>
       
    
</ul>


## Running Deep Hedging

Copied from <tt>notebooks/trainer.ipynb</tt>:

<tt>
from cdxbasics.config import Config<br>
from deephedging.trainer import train<br>
from deephedging.gym import VanillaDeepHedgingGym<br>
from deephedging.world import SimpleWorld_Spot_ATM<br>
<br>
# see print of the config below for numerous options<br>
config = Config()<br>
# world<br>
config.world.samples = 10000<br>
config.world.steps = 20<br>
config.world.black_scholes = True<br>
# gym<br>
config.gym.objective.utility = "exp2"<br>
config.gym.objective.lmbda = 10.<br>
config.gym.agent.network.depth = 3<br>
config.gym.agent.network.activation = "softplus"<br>
# trainer<br>
config.trainer.train.batch_size = None<br>
config.trainer.train.epochs = 400<br>
config.trainer.train.run_eagerly = False<br>
config.trainer.visual.epoch_refresh = 1<br>
config.trainer.visual.time_refresh = 10<br>
config.trainer.visual.pcnt_lo = 0.25<br>
config.trainer.visual.pcnt_hi = 0.75<br>
<br>
# create world<br>
world  = SimpleWorld_Spot_ATM( config.world )<br>
val_world  = world.clone(samples=1000)<br>
<br>
# create training environment<br>
gym = VanillaDeepHedgingGym( config.gym )<br>
<br>
# create training environment<br>
train( gym=gym, world=world, val_world=val_world, config=config.trainer )<br>
<br>
# print information on all available parameters and their usage<br>
print("=========================================")<br>
print("Config usage report")<br>
print("=========================================")<br>
print( config.usage_report() )<br>
config.done()<br>
</tt>

## Config Parameters

This is the output of the <tt>print( config.usage_report() )</tt> call above. It provides a summary of all config values available, their defaults, and what values where used.

<tt>
config.gym.agent.network['activation'] = softplus # Network activation function; default: relu
<br>config.gym.agent.network['depth'] = 3 # Network depth; default: 3
<br>config.gym.agent.network['width'] = 20 # Network width; default: 20
<br>config.gym.agent['agent_type'] = feed_forward #  Default: feed_forward
<br>config.gym.agent['features'] = ['price', 'delta', 'time_left'] # Named features the agent uses from the environment; default: ['price', 'delta', 'time_left']
<br>    
<br>config.gym.environment['softclip_hinge_softness'] = 1.0 # Specifies softness of bounding actions between lbnd_a and ubnd_a; default: 1.0
<br>    
<br>config.gym.objective['lmbda'] = 10.0 # Risk aversion; default: 1.0
<br>config.gym.objective['utility'] = exp2 # Type of monetary utility: mean, exp, exp2, vicky, cvar, quad; default: entropy
<br>    
<br>config.trainer.train['batch_size'] = None # Batch size; default: None
<br>config.trainer.train['epochs'] = 10 # Epochs; default: 100
<br>config.trainer.train['optimizer'] = adam # Optimizer; default: adam
<br>config.trainer.train['run_eagerly'] = False # Keras model run_eagerly; default: False
<br>config.trainer.train['time_out'] = None # Timeout in seconds. None for no timeout; default: None
<br>config.trainer.visual.fig['col_nums'] = 6 # Number of columbs; default: 6
<br>config.trainer.visual.fig['col_size'] = 5 # Plot size of a column; default: 5
<br>config.trainer.visual.fig['row_size'] = 5 # Plot size of a row; default: 5
<br>config.trainer.visual['bins'] = 200 # How many x to plot; default: 200
<br>config.trainer.visual['epoch_refresh'] = 1 # Epoch fefresh frequency for visualizations; default: 10
<br>config.trainer.visual['err_dev'] = 1.0 # How many standard errors to add to loss to assess best performance; default: 1.0
<br>config.trainer.visual['lookback_window'] = 30 # Lookback window for determining y min/max; default: 30
<br>config.trainer.visual['confidence_pcnt_hi'] = 0.75 # Upper percentile for confidence intervals; default: 0.5
<br>config.trainer.visual['confidence_pcnt_lo'] = 0.25 # Lower percentile for confidence intervals; default: 0.5
<br>config.trainer.visual['show_epochs'] = 100 # Maximum epochs displayed; default: 100
<br>config.trainer.visual['time_refresh'] = 10 # Time refresh interval for visualizations; default: 20
<br>
<br>config.world['black_scholes'] = True # Hard overwrite to use a black & scholes model with vol 'rvol' and drift 'drift; default: False
<br>config.world['corr_ms'] = 0.5 # Correlation between the asset and its mean; default: 0.5
<br>config.world['corr_vi'] = 0.8 # Correlation between the implied vol and the asset volatility; default: 0.8
<br>config.world['corr_vs'] = -0.7 # Correlation between the asset and its volatility; default: -0.7
<br>config.world['cost_p'] = 0.0005 # Trading cost for the option on top of delta and vega cost; default: 0.0005
<br>config.world['cost_s'] = 0.0002 # Trading cost spot; default: 0.0002
<br>config.world['cost_v'] = 0.02 # Trading cost vega; default: 0.02
<br>config.world['drift'] = 0.1 # Mean drift of the asset; default: 0.1
<br>config.world['drift_vol'] = 0.1 # Vol of the drift; default: 0.1
<br>config.world['dt'] = 0.02 # Time per timestep; default: One week (1/50)
<br>config.world['invar_steps'] = 5 # Number of steps ahead to sample from invariant distribution; default: 5
<br>config.world['ivol'] = 0.2 # Initial implied volatility; default: Same as realized vol
<br>config.world['lbnd_as'] = -5.0 # Lower bound for the number of shares traded at each time step; default: -5.0
<br>config.world['lbnd_av'] = -5.0 # Lower bound for the number of options traded at each time step; default: -5.0
<br>config.world['meanrev_drift'] = 1.0 # Mean reversion of the drift of the asset; default: 1.0
<br>config.world['meanrev_ivol'] = 0.1 # Mean reversion for implied vol vol vs initial level; default: 0.1
<br>config.world['meanrev_rvol'] = 2.0 # Mean reversion for realized vol vs implied vol; default: 2.0
<br>config.world['payoff'] = \<function SimpleWorld_Spot_ATM.__init__.\<locals\>.\<lambda\> at 0x0000022125590708\> # Payoff function. Parameters is spots[samples,steps+1]; default: Short ATM call function
<br>config.world['rcorr_vs'] = -0.5 # Residual correlation between the asset and its implied volatility; default: -0.5
<br>config.world['rvol'] = 0.2 # Initial realized volatility; default: 0.2
<br>config.world['samples'] = 10000 # Number of samples; default: 1000
<br>config.world['seed'] = 2312414312 # Random seed; default: 2312414312
<br>config.world['steps'] = 20 # Number of time steps; default: 10
<br>config.world['strike'] = 1.0 # Relative strike. Set to zero to turn off option; default: 1.0
<br>config.world['ttm_steps'] = 4 # Time to maturity of the option; in steps; default: 4
<br>config.world['ubnd_as'] = 5.0 # Upper bound for the number of shares traded at each time step; default: 5.0
<br>config.world['ubnd_av'] = 5.0 # Upper bound for the number of options traded at each time step; default: 5.0
<br>config.world['volvol_ivol'] = 0.5 # Vol of Vol for implied vol; default: 0.5
<br>config.world['volvol_rvol'] = 0.5 # Vol of Vol for realized vol; default: 0.5
</tt>


## Interpreting Progress Output

Here is an example of progress information printed by the <tt>NotebookMonitor</tt>:

<img src=pictures/progress.png />

The graphs show:
<ul>
<li>(1): visualizing convergence
    <ul>
        <li>(1a): last 100 epochs loss view on convergence: initial loss, full training set loss with std error, batch loss, validation loss, and the running best fit.
        </li>
        <li>(1b): loss across all epochs, same metrics as above.
        </li>
        <li>(2c): last 100 epochs Monetary utility (value) of the payoff alone, and of the hedged gains (on full training set and on validation set). Best.
        </li>
    </ul>
<li>(2) visualizing the result on the training set:
    <ul>
    <li>(2a) shows the payoff as function of terminal spot. That graph makes sense for terminal payoffs, but less so for full path dependent structures.
    </li>
    <li>(2b) shows the cash (gains) by percentile. In the example we see that the original payoff has a better payoff profile for much of the x-axis, but a sharply larger loss otherwise.
    </li>
    <li>(2c) shows the utility by percentile. This is what is optimized for.
    </li>
    </ul>
</li>
<li>(3) same as (2), but for the validation set.
</li>
<li>(4) visualizes actions:
    <ul>
        <li>(4a) shows the action per time step
        </li><li>
        (4b) shows the aggregated action as deltas accross time steps. Note that the concept of "delta"
only makes sense if the instrument is actually the same per time step, e.g. spot of an stock price. For floating options this is not a meaningful concept.
        </li>
    </ul>
</li>
    
## Misc Code Overview
    

<ul>
    <li>
        <tt>base.py</tt> contains a number of useful tensorflow utilities such as
        <ul>
                 <li><tt>tfCast, npCast</tt>: casting from and to tensorflow
            </li><li><tt>tf_back_flatten</tt>: flattens a tenosr while keeping the first 'dim'-1 axis the same.
            </li><li><tt>tf_make_dim</tt>: ensures a tensor has a given dimension, by either flattening it at the end, or adding <tt>tf.newaxis</tt>.
            </li><li><tt>mean, var, std, err</tt>: standard statistics, weighted by a density.
            </li><li><tt>mean_bins</tt>: binning by taking the average.
            </li><li><tt>mean_cum_bins</tt>: cummulative binning by taking the average.
            </li><li><tt>perct_exp</tt>: CVaR, i.e. the expecation over a percentile.
            </li><li><tt>fmt_seconds</tt>: format for seconds.
            </li>
        </ul>
    </li><li>
        <tt>world.py</tt> contains a world generator <tt>SimpleWorld_Spot_ATM</tt>.
    </li><li>
    <tt>agents.py</tt> contains an <tt>AgentFactory</tt> which is creates agents on the fly from a <tt>config</tt>.
        Only implementation provided is a simple <tt>FeedForwardAgent</tt>. Typically driven by top level <tt>config.gym.agent</tt>.
        Is also used by <tt>objectives.py</tt>.
    </li><li>
    <tt>objectives.py</tt> contains an <tt>MonetaryUtility</tt> which implements a range reasonable objectives.
    Typically driven by top level <tt>config.gym.objective</tt>.
    </li><li>
    <tt>plot_training.py</tt> contains code to provide live plots during training when running in a notebook.
    </li><li>
    <tt>afents.py</tt> contains code to provide live plots during training when running in a notebook.
    </li><li>
    <tt>gym.py</tt> contains the gym for Deep Hedging, <tt>VanillaDeepHedgingGym</tt>. It is a small script and it is recommended that every user
    reads it.
    </li>
</ul>



