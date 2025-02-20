import numpy as np
import typing
import logging
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

NUM_ACTIONS = 2

class ModelGenerator():
    def __init__(self, config: typing.Dict, features_names):

        self._features_names = features_names
        self._state_size = len(features_names)
        self._action_size = NUM_ACTIONS
        self._batch_size = config["config_model"]["batch_size"]
        self._alpha = config["config_model"]["alpha"]
        self._gamma = config["config_model"]["gamma"]
        self._epsilon = config["config_model"]["epsilon"]
        self._epsilon_min = config["config_model"]["epsilon_min"]
        self._memory_size = config["config_model"]["memory_size"]
        self._n_episodes = config["config_model"]["n_episodes"]
        self._n_steps = config["config_model"]["n_steps"]
        self._positive_reward = config["config_model"]["positive_reward"]
        self._negative_reward = config["config_model"]["negative_reward"]
        self._memory = []
        self._q_network = self.build_network()
        
        self._deep_q_learning()
        
    def build_network(self):
        return nn.Sequential(
            nn.Linear(self._state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS)
        )
    
    def policy(self, state):
        if torch.rand(1) < self._epsilon:
            return torch.randint(NUM_ACTIONS, (1,1))
        else:
            av = self._q_network(state).detach()
            return torch.argmax(av, dim = 1, keepdim=True)
            
        
        
        
        
        
    
        