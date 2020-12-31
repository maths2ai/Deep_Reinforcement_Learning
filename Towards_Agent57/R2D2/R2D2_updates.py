# Importing Section

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

from collections import deque

env = gym.make('Breakout-ram-v0')
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# Hyper-Parameters
SCORES_MEAN = 100
INPUT_SIZE = 128
HIDDEN_DIMS = (256, 128, 64)
HIDDEN_CELL_STATE_DIM = 64 
OUTPUT_SIZE = env.action_space.n
SEQ_SIZE = 80

EPSILON_FINAL = 0.001
EPSILON_INIT = 1
GAMMA = 0.99

ETA = 0.9
ALPHA = 0.9
BETA_INIT = 0.4
CAPACITY = int(1e6)
BATCH_SIZE = 128

LR = 1e-12
UPDATE_EVERY = 40
NUM_UPDATES = 1
TAU = 0.90

# Prio Buffer
class buffer(object):
    def __init__(self, capacity, batch_size, alpha):
        self.alpha = alpha
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.stores = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity), dtype = np.float32)
        
        return
    
    # TODO Add a better way to handling is_dones with a better bootstrap
    def add(self, seq, store ):
        ##print(' 2 BUFFER SEQ DIM ', len(seq))
        ##print(' 2 BUFFER store ', len(store))
        if len(self.memory)>0:
            maximum_priority = self.priorities.max()
        else:
            maximum_priority = 1.0
            
        if len(self.memory)<self.capacity:
            self.memory.append(seq)
            
            hidden_state = store[0].squeeze(0).squeeze(0)
            ##print(' 2 BUFFER hidden_state and cell state shape with double squeeze ', hidden_state.shape)
            cell_state = store[1].squeeze(0).squeeze(0)
            store = (hidden_state, cell_state)
            self.stores.append(store)
            self.priorities[self.pos] = maximum_priority
        else:
            self.memory[self.pos] = seq
            hidden_state = store[0].squeeze(0).squeeze(0)
            cell_state = store[1].squeeze(0).squeeze(0)
            store = (hidden_state, cell_state)
            self.stores[self.pos] = store
            self.priorities[self.pos] = maximum_priority
        
        self.pos = (self.pos+1)%self.capacity
        
        return
    
    def sample(self, beta):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs_temp = prios ** self.alpha
        probs = probs_temp/probs_temp.sum()
        
        indexes = np.random.choice(len(self.memory), self.batch_size, p = probs)
        
        # Weight construction
        total = len(self.memory)
        weights_temp = (total * probs[indexes]) ** (-beta)
        weights = weights_temp/weights_temp.max()
        weights = np.array(weights, dtype = np.float32)
        
        # Sequences selection
        
        batch_exp = []
        
        for idx in indexes:
            batch_exp += self.memory[idx]
        
        batch_h_c_s = [self.stores[idx] for idx in indexes]
        
        ##print(' 3 BUFFER BATCH EXP', batch_exp.shape())
        ##print(' 3 BUFFER HC STATE EXP', batch_h_c_s.shape())
        return batch_exp, batch_h_c_s, weights, indexes
    
    def memory_update (self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.memory)

# R2D2 Net

class R2D2_Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(R2D2_Net, self).__init__()
        """
        This class contains the net. Takes as inputs input_size, output_size and hidden_sizes.
        
        The forward function takes as inputs the sequence of the states, the batch_size and a seq_size
        along with hidden_states and cell_states useful to initialize the LSTM.
        
        Gives back as aoutput a series of Q_values of the form [batch_size, seq_size, number_of_actions]
        """
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        self.Linear_1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.Linear_2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.Linear_3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        
        self.batch_norm_1 = nn.BatchNorm1d(self.input_size)
        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_sizes[0])
        self.batch_norm_3 = nn.BatchNorm1d(self.hidden_sizes[1])
        self.batch_norm_4 = nn.BatchNorm1d(self.hidden_sizes[2])
        
        self.LSTM = nn.LSTM(input_size = self.hidden_sizes[2], hidden_size = self.hidden_sizes[2], num_layers = 1, batch_first = True)
        
        self.adv = nn.Linear(self.hidden_sizes[2], self.output_size)
        self.value = nn.Linear(self.hidden_sizes[2], 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, state, batch_size, seq_size, hidden_state, cell_state):
        
        state = state.view(batch_size*seq_size, -1)
        
        state = self.batch_norm_1(state)
        
        state = self.relu(self.batch_norm_2(self.Linear_1(state)))
        state = self.relu(self.batch_norm_3(self.Linear_2(state)))
        state = self.relu(self.batch_norm_4(self.Linear_3(state)))
        
        state = state.view(batch_size, seq_size, self.hidden_sizes[2])
        
        lstm_out = self.LSTM(state, (hidden_state, cell_state))
        
        hidden_state = lstm_out[1][0]
        cell_state = lstm_out[1][1]
 
        out = lstm_out[0]
        out = out.contiguous().view(batch_size * seq_size, -1)
        
        # TODO: CHECK THE DIMENSION OF THE NEXT LINE OF CODE
        adv = self.adv(out)
        value = self.value(out)
        mean = adv.mean(dim = 1).unsqueeze(1)
        
        value = value.contiguous().view(batch_size, seq_size, -1)
        adv = adv.contiguous().view(batch_size, seq_size, -1)
        mean = mean.contiguous().view(batch_size, seq_size, -1)

        # BE CAREFUL. THIS MIGHT BE WRONG IN THE UPDATING AGENT NET CONTEXT
        # CHECK ALSO AN UNSQUEEZE ON DIM 1 in adv.mean( dim = 1 ) which might be necessary
        # out_finale = value.expand(batch_size, self.output_size) + (adv - adv.mean(dim = 1).expand(batch_size, self.output_size))
        out_finale = value + (adv - mean)
        
        return out_finale, (hidden_state, cell_state)                                                                  
