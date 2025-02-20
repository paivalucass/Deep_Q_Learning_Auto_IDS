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

class Environment():
    def __init__(self, config: typing.Dict):
        self._env_index = 0
        self._env, self._env_labels = self.__build_dataset(config["dataset_train_load_paths"])

    def __build_dataset(self, paths_dictionary: typing.Dict):
        
        features_array = np.load(paths_dictionary["X_path"])
        features_array = features_array.f.arr_0      
        
        labels_array = np.load(paths_dictionary["y_path"])
        labels_array = labels_array.f.arr_0
        
        print(f"Built dataset with shape: {features_array.shape}")
        
        return features_array, labels_array
    
    def __compute_reward(self, action):
        if action == 1 and self._env_labels[self._env_index] == 1:
            return self._positive_reward
        elif action == 0 and self._env_labels[self._env_index] == 0:
            return self._positive_reward
        else:
            return self._negative_reward
        
    def __reset_env(self):
        self._env_index = 0
        
    def __step_env(self, action):
        done = False
        action = action.item()
        reward = torch.tensor(self.__compute_reward(action)).view(1, -1).float()
        self._env_index += 1
        next_state = torch.from_numpy(self._env[self._env_index]).unsqueeze(dim=0).float()
        if self._env_index == self._env.shape[0]:
            done = True
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done
    

class DQLModelGenerator():
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
        self._environment = Environment(config)
        self._q_network = self.__build_network()
        
        self._deep_q_learning()
        
    def __build_network(self):
        return nn.Sequential(
            nn.Linear(self._state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS)
        )
    
    def __policy(self, state):
        if torch.rand(1) < self._epsilon:
            return torch.randint(NUM_ACTIONS, (1,1))
        else:
            av = self._q_network(state).detach()
            return torch.argmax(av, dim = 1, keepdim=True)
            
        
        
        
        
        
    
        