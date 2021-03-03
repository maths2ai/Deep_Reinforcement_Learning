# Introduction

The DDQN algorithm is an algorithm that has been invented as an evolution of DQN algorithm.
DQN tends to overestimate results. To avoid this, in the bellman approximation.

Q(next_state, next_action)gamma + reward - Q(state, action)

the next_action is selected using a neural network, while the final value 
Q(next_state, next_action) is evaluated with another net. This decoupling procedure minimizes over-estimation.

## Requirements

