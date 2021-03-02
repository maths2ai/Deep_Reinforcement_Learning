# Introduction

This code contains my revised version of the Prioritized Experience Replay code that can be found in the repository by Higgsfield.

The idea of the code is to try to sample more often tuples of information where we can learn more.
To do this, we have to introduce compensation to make backpropagation working (otherwise, we violate some of the postulates of SGD). 
Substantially, we tend to give less weight to things we sample more often.

## Requirements

torch, math, random, numpy, matplotlib, IPython.display, gym
