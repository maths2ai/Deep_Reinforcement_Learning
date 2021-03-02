# Introduction

Noisy DQN is a DQN extension where the explorations is encoded inside the linear layers of the agent. 
A mean and a standard deviation for each weight of the layer is learned. 
The weight of the linear layer is then set equal to the sampled mean plus the sampled standard deviation which is multiplied by a random noise.

## Requirements

torch, math, random, numpy, matplotlib, IPython.display, gym
