#-*- coding:utf-8 -*-

"""
deep deterministic policy gradient
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Replay_buffer():
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        
    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
            
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)
        
    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action,capacity,device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 1e-3)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 1e-3)
        self.replay_buffer = Replay_buffer(capacity)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        
    def select_action(self, state):
        # FloatTensor 建立FloatTensor类型
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # cpu():提取CPU的data数据, numpy():tensor转numpy, flatten():降成一维
        return self.actor(state).cpu().data.numpy().flatten()
        
    def update(self,tau=0.005,batch_size=64):
        for it in range(10):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            
            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * 0.99 * target_Q).detach()
            
            # Get current Q estimate
            current_Q = self.critic(state, action)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self, directory, i):
        torch.save(self.actor.state_dict(), directory + str(i) + '_actor.pth')
        torch.save(self.critic.state_dict(), directory + str(i) + '_critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, directory, i):
        self.actor.load_state_dict(torch.load(directory + str(i) + '_actor.pth'))
        self.critic.load_state_dict(torch.load(directory + str(i) + '_critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


