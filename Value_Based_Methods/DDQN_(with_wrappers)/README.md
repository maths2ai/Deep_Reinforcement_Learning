# Introduction

The DDQN algorithm is an algorithm which has been invented as an evolution of DQN algorithm.
DQN tends to overestimate results. To avoid this, in the bellman approximation

Q(next_state, next_action)gamma + reward - Q(state, action)
