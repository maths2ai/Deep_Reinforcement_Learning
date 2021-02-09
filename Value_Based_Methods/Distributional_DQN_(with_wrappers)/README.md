# Introduction

This is my implementation of Categorical DQN. Part of the code is inspired by the Higgsfield repository on Deep Reinforcement Learning.
I will try to comment more on the common parts between my code and Higgsfield's one. 

The code has been written as an exercise for D4PG, which I need for a project.

## Requirements

torch, math, random, numpy, matplotlib, IPython.display, gym

## Notice

The code will be shortly updated with a play function (useful to record videos every k frames) and a soft update function for the Agent net.
This latter should stabilize the learning procedure.
