# Introduction

DDQN is the first improvement of the DQN algorithm. DQN tends to overestimate the value of some Q(s, a), where Q is the Q-value function.
To overcome this problem, we select the best action to choose in a state s' with the neural network Q, and then we evaluate it with the target net Q'.

## Requirements

gym, random, torch, numpy, matplotlib, IPython.display, random.

## Notice

Making video with matplotlib is computationally intensive, thus it would be beneficial not making too many of them. 
Another possibility is using the gym wrapper Monitor, gym.wrappers.Monitor.
