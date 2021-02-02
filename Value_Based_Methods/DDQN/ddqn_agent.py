# Importing section

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import random
import numpy as np

from collections import deque, namedtuple

# Device section

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Some parameters globally def
Buffer_size = int (1e5)
Batch_size = 64
Gamma = 0.99
Tau = 1e-3
Update_Every = 4
LR = 5e-4

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed , hidden_layer_1 = 256, hidden_layer_2 = 128):
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2       
   
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(self.state_size, self.hidden_layer_1)
        self.fc2 = nn.Linear(self.hidden_layer_1, self.hidden_layer_2)
        self.fc3 = nn.Linear(self.hidden_layer_2, self.action_size)
        
    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        return self.fc3(state)

class Agent():
    
    def __init__(self, env, state_space_dim, action_space_dim, seed):
        
        self.env = env
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim   
        self.seed = random.seed(seed)
        
        self.memory = ReplayBuffer(Buffer_size, Batch_size, seed)
        self.Q_network = QNetwork(self.state_space_dim, self.action_space_dim, seed).to(device)
        self.Q_network_target = QNetwork(self.state_space_dim, self.action_space_dim, seed).to(device)
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=LR)
       
        self.step = 0
        
    def act(self, state, eps):
        
        # We bring states to torch to have them in the correct format for the net. Notice also we use 
        # unsqueeze to make it into batch format
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q_network.eval()
        with torch.no_grad():
            set_of_actions = self.Q_network(state)
        self.Q_network.train()
        
        # Make everything into cpu and using copies is necessary to use numpy for argmax
        
        set_of_actions = set_of_actions.cpu().data.numpy()
        
        if random.random() <= eps:
            action = np.random.choice(np.arange(self.action_space_dim))
        else:
            action = np.argmax(set_of_actions)

        return action
    
    def Step(self, state, action, reward, next_state, is_done):
        # We add the tuple into the memory buffer, if it is the case we update
        self.memory.add(state, action, reward, next_state, is_done)
        self.step += 1
        
        self.step = (self.step + 1) % Update_Every
        if self.step == 0:
            
            if len(self.memory) > 100*Batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, Gamma)
       
        return
    
    def ddqn_selection(self, next_states, QNet):
        
        actions = QNet(next_states)
        actions = actions.cpu().data.numpy()
        actions = np.argmax(actions, axis = 1)
        actions = torch.from_numpy(actions).long().to(device).unsqueeze(1)
        return actions
    
    def learn(self, experiences, gamma):
        
        states, actions, rewards, next_states, is_dones = experiences
        # We use detach to not let it backpropagate through the net
        
        actions_for_Q_network_target = self.ddqn_selection(next_states, self.Q_network)
        Q_targets_next = self.Q_network_target(next_states).gather(1,actions_for_Q_network_target)
        Q_next_state = rewards + (gamma*Q_targets_next*(1-is_dones))
        Q_state_action = self.Q_network(states).gather(1, actions)
        
        loss = F.mse_loss(Q_state_action, Q_next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.Q_network, self.Q_network_target, Tau)
        
        return
        
    def soft_update(self, model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau*local_param.data +(1.0 - tau)*target_param.data)
        return
    
    def play(self):
        
        state = self.env.reset()
        sequence_of_frames = []
        
        while True:
            sequence_of_frames.append(self.env.render(mode='rgb_array'))
            state = torch.from_numpy(state).float().to(device)
            actions = self.Q_network(state)
            actions = actions.detach().cpu().numpy()
            action = np.argmax(actions)
            new_state, reward, is_done, info = self.env.step(action)
            state = new_state
        
            if is_done == True:
                return sequence_of_frames 
        
class ReplayBuffer():
    
    def __init__(self, Buffer_dim, Batch_size, seed):
        
        self.Buffer_dim = Buffer_dim
        self.Batch_size = Batch_size
        self.Buffer = deque(maxlen = self.Buffer_dim)
        self.experience = namedtuple('Experience', field_names = ['state', 'action', 'reward', 'next_state', 'is_done'])
        self.seed = seed
        
    def add(self, state, action, reward, next_state, is_done):
        
        e = self.experience(state, action, reward, next_state, is_done)
        self.Buffer.append(e)
        return
    
    def sample(self):
        
        experiences = random.sample(self.Buffer, k = self.Batch_size)
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(device)
        is_dones = torch.from_numpy(np.vstack([exp.is_done for exp in experiences if exp is not None])).int().float().to(device)
        
        return (states, actions, rewards, next_states, is_dones)
    
    def __len__(self):
        
        return len(self.Buffer)
        
        
                         
     