# Introduction

The PPO algorithm is a modification of the Reinforce algorithm.
It improves some limitations of the Reinforce algorithm. Specifically instead of trying to maximaze the function:

$$\sum_{episode = 1}^{n} \sum_{t = 0}^{T} Reward_{episode}log(p(a_t|s_t),$$

it maximizes a different function which is built to avoid the following problematics:

1. Not wasting trajectories (Reinforce just trashes a trajectory once used for improving the policy)
2. Addresses the problem of giving the right weight to actions that bring more reward (Reinforce tends to "emphasize" the same all the actions connected with a positive episode reward)

The function has the shape:

$$\sum_{episode = 1}^{n} \sum_{t = 0}^{T} Reward-Collect-From-Time-t-Onwards_{episode}\frac{p(a_t|s_t)_{\mbox{under new policy}}{p(a_t|s_t)_{\mbox{under old policy}.$$

Notice that substituting a function with another one introduces an error. This error might be catastrophic. To avoid catastrophies we clip the previous function.

## Requirements

torch,
gym,
numpy,
JSAnimation.IPython_display,
matplotlib,
IPython.display,
multiprocessing.

