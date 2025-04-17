import numpy as np
import typing
import logging
import torch
import copy
import time
import random
import logging
import statistics
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

NUM_ACTIONS = 2
LOG_FILE_PATH = "/home/slurm/pesgradivn/lcap/Deep_Q_Learning_Auto_IDS/output/metrics.log"

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
        
    def sample(self): # Modifed the sampling from random to sequential
        assert self.can_sample()
        start_idx = random.randint(self._batch_size, len(self.memory) - self._batch_size)
        batch = self.memory[start_idx:start_idx + self._batch_size]
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]

    def can_sample(self):
        return len(self.memory) >= self._batch_size * 10
        
    def __len__(self):
        return len(self.memory)

class Environment():
    def __init__(self, config: typing.Dict, dataset: typing.Dict):
        self._env_index = 0
        self._env, self._env_labels = self.__build_dataset(dataset)
        self._positive_reward = config["config_model"]["positive_reward"]
        self._negative_reward = config["config_model"]["negative_reward"]

    def __build_dataset(self, paths_dictionary: typing.Dict):
        
        features_array = np.load(paths_dictionary["X_path"])
        features_array = features_array.f.arr_0      
        
        labels_array = np.load(paths_dictionary["y_path"])
        labels_array = labels_array.f.arr_0
        
        if len(features_array) > 1000000:
            pass
        
        else:
            pass
        
        print(f"Built dataset with shape: {features_array.shape}")
        
        return features_array, labels_array
    
    def __compute_reward(self, action):
        
        if action == 1 and self._env_labels[self._env_index] == 1:
            return self._positive_reward
        
        elif action == 0 and self._env_labels[self._env_index] == 0:
            return self._positive_reward
        
        else:
            return self._negative_reward
        
    def reset_env(self):
        self._env_index = 0
        return torch.from_numpy(self._env[0]).unsqueeze(dim=0).float()
        
    def step_env(self, action):
        done = 0
        action = action.item()
        reward = torch.tensor(self.__compute_reward(action)).view(1, -1).float()
        self._env_index += 1
        next_state = torch.from_numpy(self._env[self._env_index]).unsqueeze(dim=0).float()
        if self._env_index == self._env.shape[0] - 1:
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
        self._n_episodes_train = config["config_model"]["n_episodes_train"]
        self._n_episodes_test = config["config_model"]["n_episodes_test"]
        self._n_steps = config["config_model"]["n_steps"]
        self._replay_memory = ReplayMemory(config)
        self._environment_train = Environment(config, config["dataset_train_load_paths"])
        self._environment_test = Environment(config, config["dataset_test_load_paths"])
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(
        filename=LOG_FILE_PATH,  
        filemode='a',         
        format='%(asctime)s - %(levelname)s - %(message)s', 
        level=logging.INFO)
        self.q_network = self.__build_network()
        self._target_q_network = copy.deepcopy(self.q_network).eval()
        self._start_time = 0
        self._end_time = 0
        self._c_report = None
        self._confusion_matrix = None
            
    def __build_network(self):
    return nn.Sequential(
        nn.Linear(self._state_size, 128),
        nn.BatchNorm1d(128),  # Normalize activations
        nn.ReLU(),
        nn.Dropout(0.3),  # Regularization
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),        
        nn.Linear(64, NUM_ACTIONS)
    )
    
    def __policy(self, state):
        if torch.rand(1) < self._epsilon:
            return torch.randint(NUM_ACTIONS, (1,1))
        else:
            av = self.q_network(state).detach()
            return torch.argmax(av, dim = 1, keepdim=True)
    
    def __policy_test(self, state):
        av = self.q_network(state).detach()
        return torch.argmax(av, dim = 1, keepdim=True)
        
    def _deep_q_learning(self):
        """ Initialize Neural Network Optimizer """

        optim = AdamW(self.q_network.parameters(), lr=self._alpha)
        stats = {'MSE Loss': [], 'Returns': []}

        # For loss plateau detection
        recent_losses = []
        loss_window_size = 100
        std_threshold = 1e-4  

        for episode in tqdm(range(1, self._n_episodes_train + 1)):
            
            state = self._environment_train.reset_env()
            done = False
            ep_return = 0

            while not done:
                """
                This loop will pass over the whole dataset until it "finishes" passing throught all states
                """
                # Calculate an action to be taken, based on the policy output
                action = self.__policy(state)
                # Apply the action at the enviroment and returns new state and reward 
                next_state, reward, done = self._environment_train.step_env(action)
                # Insert the transition in the replay memory so it can be used by the Q-Network 
                self._replay_memory.insert([state, action, reward, done, next_state])
                
                # Verify if the replay memory has enough samples to be batched 
                if self._replay_memory.can_sample():
                    # Takes a batch of transitions 
                    state_batch, action_batch, reward_batch, done_batch, next_state_batch = self._replay_memory.sample()
                    # Pass the batches through the neural network and collects only the q-values taken by the policy 
                    qsa_b = self.q_network(state_batch).gather(1, action_batch)
                    
                    # Calculate the q-value for the next state batch 
                    next_qsa_b = self._target_q_network(next_state_batch)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

                    # Calculate the target q-value batch 
                    target_b = reward_batch + (1 - done_batch) * self._gamma * next_qsa_b
                    # Calculate the mean squared error (loss function)
                    loss = F.mse_loss(qsa_b, target_b)
                    self.q_network.zero_grad()
                    # Do the backpropagation 
                    loss.backward()
                    # Update the neural network weights and biases
                    optim.step()

                    loss_val = loss.item()
                    stats['MSE Loss'].append(loss_val)
                    recent_losses.append(loss_val)

                    # Check loss stability over last 100 steps
                    if len(recent_losses) >= loss_window_size and episode > 5:  # Require minimum episodes
                        std_dev = statistics.stdev(recent_losses)
                        if std_dev < std_threshold:
                            print(f"Stopping early at episode {episode}: loss not changing (std={std_dev:.6f})")
                            return stats
                        recent_losses.pop(0)

                state = next_state
                ep_return += reward.item()

            stats["Returns"].append(ep_return)

            # Decay epsilon
            self._epsilon = max(self._epsilon_min, self._epsilon * 0.995)
            
            if episode % 10 == 0:
                self._target_q_network.load_state_dict(self.q_network.state_dict())

        return stats        
    
    def test_model(self):
        y_true = []
        y_pred = []

        state = self._environment_test.reset_env() 
        done = False
        
        self._start_time = time.time()

        while not done:
            action = self.__policy_test(state) 
            
            y_true.append(self._environment_test._env_labels[self._environment_test._env_index])  
            y_pred.append(action.item())
            next_state, reward, done = self._environment_test.step_env(action)
            state = next_state
            
        self._end_time = time.time()

        print("Classification Report:")
        self._c_report = classification_report(y_true, y_pred, target_names=["Intrusion", "Normal"], output_dict=True)
        print(self._c_report)
        
        print("\nConfusion Matrix:")
        self._confusion_matrix = confusion_matrix(y_true, y_pred)
        print(self._confusion_matrix)
        
    def save_metric(self, config: typing.Dict):
        
        data_config = config["config"]
        model_config = config["config_model"]
        
        elapsed_time = self._end_time - self._start_time

        log = ""
            
        log = f""" ALGORITHM: {model_config["algorithm"]} 
                            DATA: {data_config["labeling_schema"]}
                            BATCH SIZE: {model_config["batch_size"]} 
                            ALPHA: {model_config["alpha"]} 
                            GAMMA: {model_config["gamma"]} 
                            EPSILON: {model_config["epsilon"]} 
                            EPSILON_MIN: {model_config["epsilon_min"]} 
                            MEMORY_SIZE: {model_config["memory_size"]} 
                            NUMBER OF EPISODES TRAIN: {model_config["n_episodes_train"]}
                            NUMBER OF STEPS: {model_config["n_steps"]} 
                            POSITIVE REWARD: {model_config["positive_reward"]}
                            NEGATIVE REWARD: {model_config["negative_reward"]}
                            ATTACK TRAIN: {model_config["attack_train"]}
                            ATTACK TEST: {model_config["attack_test"]}
                            POLICY_TEST: {model_config["policy_test"]}
                            ELAPSED_TIME: {elapsed_time} Seconds
                            TIME_PER_SAMPLE: {elapsed_time/len(self._environment_test._env_labels)} Seconds
                            | GENERAL ACCURACY {self._c_report['accuracy']} | PRECISION NORMAL {self._c_report['Normal']['precision']} | RECALL NORMAL {self._c_report['Normal']['recall']} | F1-SCORE NORMAL {self._c_report['Normal']['f1-score']} | PRECISION ANOMALY {self._c_report['Intrusion']['precision']} | RECALL ANOMALY {self._c_report['Intrusion']['recall']} | F1-SCORE ANOMALY {self._c_report['Intrusion']['f1-score']} |"""
        
        self._logger.info(log)
        
        log = self._confusion_matrix
        
        self._logger.info(log)