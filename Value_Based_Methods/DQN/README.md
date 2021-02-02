# Introduction 

This folder contains the DQN algorithm.
This algorithm uses a neural network to learn a Q-table function.
To make bootstrap possible, in the Bellman equation, a target Q-table is created and periodically updated.

## Requirements

gym, random, torch, numpy, matplotlib, IPython.display,random

## Notice

Making video with matplotlib is computationally intensive, thus it would be beneficial not making too many of them. 
Another possibility is using the gym wrapper Monitor, gym.wrappers.Monitor.
