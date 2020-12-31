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
    
# Agent
class agent(): 
    def __init__(self, input_size, output_size, hidden_sizes):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        self.net = R2D2_Net(self.input_size, self.output_size, self.hidden_sizes).to(device)
        self.net_tg = R2D2_Net(self.input_size, self.output_size, self.hidden_sizes).to(device)
        
        self.hard_update(self.net, self.net_tg)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = LR)
        
        return
    
    def act(self, state, hidden_state, cell_state, epsilon):
        
        state = torch.FloatTensor(state).view(1, 1, self.input_size).to(device)
                
        self.net.eval()
        with torch.no_grad():
            Q_values, (hidden_state, cell_state) = self.net(state, 1, 1, hidden_state, cell_state)
                    
        self.net.train()
        
        if random.random() > epsilon:
            # TODO (DURING THE RUN) CHECK IF THE DIMENSIONS OF ACTION ARE CORRECT
            action = torch.argmax(Q_values, dim = 2).squeeze(0).cpu().data.numpy()
            #print(action.dtype)
            ##print('1 ACT AGENT GREEDY', action.shape)
        else:
            action = [random.choice(range(self.output_size))]
            #print('1 ACT AGENT RANDOM', action.shape)

        return action, (hidden_state, cell_state)
    
        ### NOTICE THIS HAS TO BE RLLY SERIOUSL CHECKED. AS THE EVALUATE FUNCTION
    def update(self, batch_exp, batch_h_c_s, weights, beta, indices):
        hidden_states, cell_states = zip(*batch_h_c_s)
        states, actions, rewards = zip(*batch_exp)
        ##print(' 4 UPDATES hidden_states shape after zip ', hidden_states.shape())
        hidden_states = torch.cat(hidden_states)
        ##print(' 4 UPDATES hidden_states shape after zip and torch.cat ', hidden_states.size())
        cell_states = torch.cat(cell_states)
        hidden_states = hidden_states.view(1, BATCH_SIZE, HIDDEN_CELL_STATE_DIM).to(device)
        cell_states = cell_states.view(1, BATCH_SIZE, HIDDEN_CELL_STATE_DIM).to(device)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)

        weights = torch.FloatTensor(weights).to(device)
        states = states.view(BATCH_SIZE * SEQ_SIZE * self.input_size)
        ##print(' 4 UPDATES states shape after states.view(BATCH_SIZE * SEQ_SIZE * self.input_size)', states.size())


        rewards = np.asarray(rewards)
        rewards = rewards.reshape(BATCH_SIZE, SEQ_SIZE, -1)
        ##print(' 4 UPDATES rewards shape after rewards.view(BATCH_SIZE * SEQ_SIZE, -1 )', rewards.shape)
        
        # Net Update
        
        #  TO DO CHECK ALL THE POSSIBLE CASTING PROBLEMS
            
        Q_values, (hiddens_Q_values, cell_Q_values) = self.net(states, BATCH_SIZE, SEQ_SIZE, hidden_states, cell_states)
        # TO DO. DOUBLE CHECK THIS
        actions = actions.view(BATCH_SIZE, SEQ_SIZE, -1)
        ##print(' 4 UPDATES actions shape after actions.view(BATCH_SIZE * SEQ_SIZE )', actions.size())

        Q_values = Q_values.gather(dim = 2, index = actions)
        Q_values = Q_values[:, 40:-1, :]
        ##print(' 4 UPDATES Q_values shape ', Q_values.size())
        
        
        Q_next_values, (hiddens_Q_values, cell_Q_values) = self.net_tg(states, BATCH_SIZE, SEQ_SIZE, hidden_states, cell_states)

        ##print('4 UPDATES Q_next_values shape', Q_next_values[:,-1,:].size())

        Q_next_values = torch.max(Q_next_values[:,-1,:], dim = 1)[0]

        Q_targets = self.bootrstap_rewards(rewards, Q_next_values)

        td = (Q_targets.detach() - Q_values).pow(2).sum(dim = 1) * weights
        
        ## NEXT THREE LINES SHOULD BE USED FOR PARALLELISM
        delta_max = torch.max(td, dim = 1)[0]
        delta_average = torch.mean(td, dim = 1)
        prios = 0.9 * delta_max + (1-0.9) * delta_average + 1e-5
        
        loss = td.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        buffer.memory_update(indices  , prios.data.cpu().numpy())
        self.soft_update(self.net, self.net_tg, TAU)
        
        return loss
        
        # NOTICE THIS MIGHT NOT BE NECESSARY CHECK OUT 
        # MAYBE IS_DONES, HIDDEN_STATES and CELL_STATES????
    #def evaluate(self, sequence, hidden_state, cell_state ):
     #   states, actions, rewards = zip(*sequence)
        
        # NOTICE: HERE IT IS VERY LIKELY WE NEED AN UNSQUEEZE
        # FOR STATES, ACTIONS (DURING GATHER) AND REWARDS FOR CASTING PROBLEMS
      #  states = torch.FloatTensor(states).to(device)
       # actions = torch.LongTensor(actions).to(device)
       # rewards = torch.FloatTensor(rewards).to(device)
        # MAYBE THIS IS UN-NECESSARY WRITE MAIN AND CHECK 
       # hidden_state = hidden_state.view(1, 1, -1)
       # cell_state = cell_state.view(1, 1, -1)
        
        # Q_values
        
        # NOTICE: THIS SHOULD BE CORRECT BUT THERE MIGHT BE A CASTING PROBLEM.
        # CHECK THIS PART.
       # actions = actions.view(BATCH_SIZE, SEQ_SIZE, -1)
       # actions = actions[:,40:-1,:]
       # Q_values = self.net(states[:-1], 1, SEQ_SIZE - 1, hidden_state, cell_state)
       # Q_values = Q_values[40:]
       # Q_values = Q_values.gather(dim = 2, index = actions)
        
        # Q_next
        
        # NOTICE: THIS IS DONE WITH THE ADD OF DDQN TECHNIQUE
        # CHECK THIS PART
        #Q_next_values, _, _ = self.net(states, BATCH_SIZE, SEQ_SIZE, hidden_state, cell_state)
        #Q_next_values = Q_next_values[41:]
        #new_actions = torch.argmax(Q_next_values, dim = 2)
        
        #Q_next_values_tg, _, _ = self.net_tg(states, BATCH_SIZE, SEQ_SIZE, hidden_state, cell_state)
        #Q_next_values_tg = Q_next_values_tg[41:]
        #Q_next_values_tg = Q_next_values.gather(dim = 2, new_actions)
        # NOTICE: IT IS UNCLEAR IF GAMMA HAS TO BE DISCOUNTED OR NOT AND, IN THIS CASE, WHEN
        # IT HAS TO BE STARTED TO COMPUTED
        
        ## NOTICE: IT IS UNCLEAR HOW TO WRITE IS_DONES
        #Q_targets = rewards[40:79] + GAMMA * Q_next_values_tg() * (1.0 - is_dones)
        
        #td = (Q_targets - Q_values).pow(2)
        #delta_max = td.max()
        #delta_sum = td.mean()
        #prio = 
        
    def bootrstap_rewards(self, rewards, last_bootstrapping_value):
        lista_bootstrap = []
        last_bootstrapping_value = last_bootstrapping_value.detach().cpu().numpy()
        ##print('5 BOOTSTRAP REWARDS last Q(s,a) size', last_bootstrapping_value.shape)
        
        for i in range(BATCH_SIZE):
            temp = last_bootstrapping_value[i]
            #print(rewards[i].shape)
            rewards_reversed = np.flip(rewards[i])
            lista_temp_bootstrap = []
            #print(rewards_reversed)
            #print(rewards_reversed.shape)
            
            for j in range(1,40):
                temp = GAMMA * temp
                temp += rewards_reversed[j] 
                lista_temp_bootstrap.append(temp)
            
            lista_temp_bootstrap = np.flip(np.asarray(lista_temp_bootstrap))
            ##print(' 5 BOOTSTRAP REWARDS Single reward bootstrap flipped ', lista_temp_bootstrap.shape)
            lista_bootstrap.append(lista_temp_bootstrap)
    
        lista_bootstrap = torch.FloatTensor(lista_bootstrap).to(device)
        ##print('5 BOOTSTRAP REWARDS list of list shape ', lista_bootstrap.size())
        return lista_bootstrap
           
    def soft_update(self, network, target_network, tau):
        
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, network, target_network):
        target_network.load_state_dict(network.state_dict())
    
    ## TODO ADD THE VIDEO
    def play(self):
        state = env.reset()
        score = 0
        hidden_state = torch.zeros(HIDDEN_CELL_STATE_DIM).view(1, 1, HIDDEN_CELL_STATE_DIMENSION)
        cell_state = torch.zeros(HIDDEN_CELL_STATE_DIM).view(1, 1, HIDDEN_CELL_STATE_DIMENSION)
        
        while True:
            action, hidden_state, cell_state = self.act(state, hidden_state, cell_state)
            next_state, reward, is_done, _ = env.step(action)
            score += reward
            state = next_state
            
            if is_done == True:
                return score
 
# Support functions to agent and buffer. Epsilon Decay
def epsilon_decay(numbers, epsilon_init, epsilon_final):
    return epsilon_final + (epsilon_init-epsilon_final)/np.log(1+(np.log(1+numbers)))**3

def beta_decay(numbers, beta_init):
    return min(1.0, beta_init + numbers * (1.0 - beta_init))

# Instatiating objects
buffer = buffer(CAPACITY, BATCH_SIZE, ALPHA)
agent = agent(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_DIMS)

def main(num_episodes):    
    
    counter = 1 # Needed for epsilon decay, beta decay
    scores = []
    scores_deque = deque(maxlen = SCORES_MEAN) # Useful to check if we reached the goal.
    losses = []
    loss_tracking = deque(maxlen = 40)
    hidden_layers_deque = deque(maxlen = 80)
    sequence_deque = deque(maxlen = 80)
    best_value = -np.inf
    flag = 0
       
    for episode in range(1, num_episodes + 1):
        counter_for_episode = 1
        state = env.reset()
        score = 0
        # NOTICE this is needed since the first time one needs to reshape the hidden_state and cell_state under the proper dimensions
        hidden_state = torch.zeros(HIDDEN_CELL_STATE_DIM).view(1, 1, HIDDEN_CELL_STATE_DIM).to(device)
        cell_state = torch.zeros(HIDDEN_CELL_STATE_DIM).view(1, 1, HIDDEN_CELL_STATE_DIM).to(device)
        
        while True:
            hidden_layers_deque.append((hidden_state, cell_state))
            if flag == 1:
                counter += 1
            epsilon = epsilon_decay(counter, EPSILON_INIT, EPSILON_FINAL)
            action, (hidden_state, cell_state) = agent.act(state, hidden_state, cell_state, epsilon)
            next_state, reward, is_done, _ = env.step(action)
            
            sequence_deque.append((state, action, reward))
            score += reward
            counter_for_episode += 1
            
            
            if is_done and counter_for_episode <80:
                scores.append(score)
                scores_deque.append(score)
                print('The score for episode {} is {}'.format(episode, score))
                print('counter for ep ', counter_for_episode)
                break
            elif is_done and counter_for_episode >=80:
                # One should check out if the deque creates problem later
                scores.append(score)
                scores_deque.append(score)
                buffer.add(sequence_deque, hidden_layers_deque[0])
                print('The score for episode {} is {}'.format(episode, score))
                print('counter for ep ', counter_for_episode)
                break
            elif(counter_for_episode % 40 == 0) and counter_for_episode >= 80:
                print('episode added element {}', counter_for_episode)
                buffer.add(sequence_deque, hidden_layers_deque[0])
                
            if len(buffer) > 10*BATCH_SIZE: 
                flag == 1
                if counter % UPDATE_EVERY == 0:
                    for update in range(NUM_UPDATES):
                        ##print('update')
                        beta = beta_decay(counter, BETA_INIT)
                        batch_exp, batch_h_c_s, weights, indices = buffer.sample(beta)
                        loss = agent.update(batch_exp, batch_h_c_s, weights, beta, indices)
                        loss_tracking.append(loss.cpu().detach().numpy())
                        losses.append((episode, loss))
            
            # TODO Improve the next if using a deque for losses as well and giving back a mean of the losses
            if (counter_for_episode % 40 == 0) and (counter > 2*BATCH_SIZE) :
                temp_losses = np.mean(np.asarray(loss_tracking))
                print('The loss is {}'.format(loss))
                    
            state = next_state
        
        if episode % 10 == 0:
            temp_score = scores[-10:]
            temp_score = np.asarray(temp_score)
            print('The average score of the last ten episodes is ', temp_score.mean())
        
        if episode % SCORES_MEAN == 0:
            mean = np.asarray(scores_deque).mean()
            print('The average of the last {} is {}'.format(SCORES_MEAN, mean))    
            
            # TODO To modify to see what is the best way of saving
            if best_value < mean:
                best_value = mean
                #torch.save({'Q': agent.net, 'Q_tg': agent.net_tg, 'buffer': buffer}, 'Saving/dict.pth')
