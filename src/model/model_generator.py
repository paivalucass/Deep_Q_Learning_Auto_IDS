import numpy as np
import typing
import logging
import torch
import copy
import random
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

NUM_ACTIONS = 2

class ReplayMemory():
    def __init__(self, config: typing.Dict):
        self.capacity = config["config_model"]["replay_capacity"]
        self._batch_size = config["config_model"]["batch_size"]
        self.memory = []
        self.position = 0

    
    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]
        
    def can_sample(self):
        return len(self.memory) >= self._batch_size * 10
        
    def __len__(self):
        return len(self.memory)

class Environment():
    def __init__(self, config: typing.Dict):
        self._env_index = 0
        self._env, self._env_labels = self.__build_dataset(config["dataset_train_load_paths"])
        self._positive_reward = config["config_model"]["positive_reward"]
        self._negative_reward = config["config_model"]["negative_reward"]

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
        return self._env[0]
        
    def __step_env(self, action):
        done = 0
        action = action.item()
        reward = torch.tensor(self.__compute_reward(action)).view(1, -1).float()
        self._env_index += 1
        next_state = torch.from_numpy(self._env[self._env_index]).unsqueeze(dim=0).float()
        if self._env_index == self._env.shape[0]:
            done = 1
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done

class DQLModelGenerator():
    def __init__(self, config: typing.Dict, features_names):

        self._features_names = features_names
        self._state_size = len(features_names)
        self._action_size = NUM_ACTIONS
        self._alpha = config["config_model"]["alpha"]
        self._gamma = config["config_model"]["gamma"]
        self._epsilon = config["config_model"]["epsilon"]
        self._epsilon_min = config["config_model"]["epsilon_min"]
        self._memory_size = config["config_model"]["memory_size"]
        self._n_episodes = config["config_model"]["n_episodes"]
        self._n_steps = config["config_model"]["n_steps"]
        self._replay_memory = ReplayMemory(config)
        self._environment = Environment(config)
        self._target_q_network = copy.deepcopy(self._q_network).eval()
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
            
    def _deep_q_learning(self):
        """ Initialize Neural Network Optimizer """
        
        optim = AdamW(self._q_network.parameters(), lr=self._alpha)
        stats = {'MSE Loss': [], 'Returns': []}

        """ Episode Loop
        
        Every episode is a run over the whole dataset. (while not done) """
        
        for episode in tqdm(range(1, self._n_episodes + 1)):
                        
            # Reset the environment every time it starts a new episode 
            state = self._environment.__reset_env()
            done = False
            ep_return = 0

            while not done:
                """
                This loop will pass over the whole dataset until it "finishes" passing throught all states
                """
                # Calculate an action to be taken, based on the policy output
                action = self.__policy(state)
                # Apply the action at the enviroment and returns new state and reward 
                next_state, reward, done = self._environment.__step_env(action)
                # Insert the transition in the replay memory so it can be used by the Q-Network 
                self._replay_memory.insert([state, action, reward, done, next_state])
                
                # Verify if the replay memory has enough samples to be batched 
                if self._replay_memory.can_sample():
                    # Takes a batch of transitions 
                    state_batch, action_batch, reward_batch, done_batch, next_state_batch = self._replay_memory.sample()
                    # Pass the batches through the neural network and collects only the q-values taken by the policy 
                    qsa_b = self._q_network(state_batch).gather(1, action_batch)
                    
                    # Calculate the q-value for the next state batch 
                    next_qsa_b = self._target_q_network(next_state_batch)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

                    # Calculate the target q-value batch 
                    target_b = reward_batch + ~done_batch * self._gamma * next_qsa_b
                    # Calculate the mean squared error (loss function)
                    loss = F.mse_loss(qsa_b, target_b)
                    self._q_network.zero_grad()
                    # Do the backpropagation 
                    loss.backward()
                    # Update the neural network weights and biases
                    optim.step()

                    stats['MSE Loss'].append(loss.item())
                    
                state = next_state
                ep_return += reward.item()
                
            stats["Returns"].append(ep_return)
            
            if episode % 10 == 0:
                self._target_q_network.load_state_dict(self._q_network.state_dict())

        return stats        