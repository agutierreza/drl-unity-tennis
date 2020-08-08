import numpy as np
import random
import copy
from collections import namedtuple, deque

#from model import Actor, Critic
#These will be for the new actor and critic agent classes
from actoragent import ActorAgent
from criticagent import CriticAgent

from noise import OUNoise
from replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentCommon():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Noise process
        #self.noise = OUNoise(action_size, random_seed)
        self.noise = OUNoise((self.num_agents, action_size), seed = random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        #define 2 actors (for each side of the court) and a (shared) critic
    
    def step(self, state, action, reward, next_state, done):
        #call step for each actor
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            pass
            #collect experience sample and train critic based on actors


    def act(self, state, add_noise=True):
        pass
        #call 'act' for both actors and retreive actions
        #return[actionL,actionR]
    
    def reset(self):
        self.noise.reset()


