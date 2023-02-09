# Deep Hedging - Agent Architecture
### Description of the default Deep Hedging Agent 

This document describes the default agent used in the Deep Hedging code base. This is _not_ the agent used in our [Deep Hedging paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3120710) but a more advanced version. 


The Deep Hedging problem for a horizon $T$ hedged over $M$ time steps with $N$ hedging instruments is finding an optimal *action* function $a$ as a function of feature states $s_0,\ldots,s_{T-1}$ which solves
$$
 \sup_a:\ \mathrm{E}\left[\ 
    Z_T + \sum_{t=0}^{T-1} a(s_t) \cdot DH_t + \gamma_t \cdot | a(s_t) H_t |
 \ \right] \ .
$$ c.f. the description in the [main document](README.md).

The _agent_ here is the functional $a$ which maps the current **state** $s_t$ to an **action**, which in this case is simply how many units of the $n$ hedging instruments to buy in $t$. In practise, $s_t$ represents the features available at time $t$.  

The agent provided in ``agent.py`` provides both a "recurrent" and a non-recurrent version, but it should be noted that since the state at time $s_t$ contains the previous steps action $a_{t-1}$ as well as the aggregate position $\delta_{t-1}$ strictly speaking even a "non-recurrent" agent is actually recurrent.

Define the function
$$
    \mathbf{1}(x) := \left\{ \begin{array}{ll} 1 & \mathrm{if}\ \mathrm{tanh}(x) > 0.\\
                                    0. & \mathrm{else} 
                                    \end{array}\right.
$$ with values in $\{0,1\}$.


* **Classic** (recurrent) **States** are given by
$$
   h_t = F(s_t, h_{t-1}) 
$$ where $F$ is a neural network. This is the original recurrent network formulation and suffers from both gradient explosion and long term memory loss. To alleviate this somewhat we restrict $h_t$ to $(-1,+1)$.

* **Aggregate States** represent aggregate statistics of the path, such as realized vol, skew or another functions of the path. The prototype exponential aggregation function for such a hidden state $h$ is given as 
$$
   h_t = h_{t-1} (1 - z_t ) + z_t F(s_t, h_{t-1})  \ \ \ \ z_t \in [0,1]
$$ where $F$ is a neural network, and where $z_t$ is an "update gate vector". This is also known as a "gated recurrent unit" and is similar to an LSTM node. 
In quant finance such states are often written in diffusion notation with $z_t \equiv \kappa_t\, dt$ where $\kappa_t\geq 0$ is a mean-reversion speed. In this case the $dt\downarrow 0$ limit becomes
$$
    h_t = e^{-\int_0^t\! \kappa_s\,ds} h_0 + \int_0^t\!\! \kappa_u e^{-\int_u^t\! \kappa_s\,ds} F(s_u,h_{u-})\,du \ .
$$ The appendix provides the derivation of this formula from its associated SDE.

* **Past Representation States:** information gathered at a fixed points in time, for example the spot level at a reset date. Such data are not accessible before their observation time.
The prototype equation for such a process is
$$
 h_t = h_{t-1} (1 - z_t ) + z_t F(s_t, h_{t-1}) \ \ \ \ z_t \in \{0,1\}
 $$ which looks similar as before but where $z_t$ now only takes values in $\{0,1\}$. This allows encoding, for example, the spot level at a given fixed time $\tau$.

* **Event States** which track e.g. whether a barrier was breached. This looks similar to the previous example, but the event itself has values in $\{0,1\}$.
$$
 h_t = h_{t-1} (1 - z_t ) + z_t \mathbf{1}\!\Big(  F(s_t, h_{t-1}) \Big)  \ \ \ \ z_t \in \{0,1\}
 $$  where we need to make sure that $h_{-1}\in\{0,1\}$, too. 
 

## Appendix: Mean Reversion Math

#### Constant mean reversion
Consider
$$
    dx_t = \kappa (y_t - x_t)\,dt
    \ \ \ \ \Leftrightarrow \ \ \ \
    x_{t+dt} = \kappa dt\ y_t + (1 - \kappa dt )\ x_t
$$ Then
$$
    d\left( e^{\kappa t} x_t \right)
        = \kappa e^{\kappa t} x_t\,dt +
        \kappa e^{\kappa t}(y_t - x_t)\,dt = \kappa e^{\kappa t}y_t\,dt
$$ and therefore
$$
    x_t = e^{-\kappa t} x_0 + \kappa \int_0^t e^{-\kappa(t-s)} y_s\,ds
$$
#### Mean reversion as function of time

Let
$$
dx_t = \kappa_t (y_t - x_t)\,dt
$$ Set $K_t:=\exp(\int_0^t \kappa_s\,ds)$. Then
$$
    d\left( K_t x_t \right)
        = \kappa_t K_t x_t\,dt +
        \kappa_t K_t (y_t - x_t)\,dt = \kappa_t K_t y_t\,dt
$$ and therefore
$$
    x_t = e^{-\int_0^t\! k_s\,ds} x_0
        + \int_0^t\! \kappa_r\,e^{-\int_r^t\!\kappa_s\,ds } y_s\,ds
$$