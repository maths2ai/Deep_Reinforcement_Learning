# Introduction  

Cross-Entropy Method is a policy method that learns from the most profitable trajectories.
It orders the trajectories by the magnitude of rewards, selects the top k percentile of them, 
and learns what actions these k-percentile trajectories performed under the states s.

The code is largely inspired by the book "Deep Reinforcement Learning Hands On".

## Requirements

gym,
torch,
numpy.
