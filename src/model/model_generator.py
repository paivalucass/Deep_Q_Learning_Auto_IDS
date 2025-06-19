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
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import wandb
import pandas as pd

NUM_ACTIONS = 2
LOG_FILE_PATH = "/home/slurm/pesgradivn/lcap/Deep_Q_Learning_Auto_IDS/output/metrics.log"

class ReplayMemory():
    def __init__(self, config: typing.Dict):
        self._capacity = config["config_model"]["replay_capacity"]
        self._batch_size = config["config_model"]["batch_size"]
        self.memory = []
        self.position = 0
    
    def insert(self, transition):
        if len(self.memory) < self._capacity:
            self.memory.append(None)
        
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self._capacity 
        
    # Random Sampling 
    def sample(self): 
        assert self.can_sample()
        batch = random.sample(self.memory, self._batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch] 

    def can_sample(self):
        return len(self.memory) >= self._batch_size * 10
        
    def __len__(self):
        return len(self.memory)

class Environment():
    def __init__(self, config: typing.Dict, dataset: typing.Dict, dataset_type = "train"):
        self._env_index = 0
        self._positive_reward = config["config_model"]["positive_reward"]
        self._negative_reward = config["config_model"]["negative_reward"]
        self._start_train = config["config_model"]["start_dataset_train"]
        self._end_train = config["config_model"]["end_dataset_train"]
        self._start_test = config["config_model"]["start_dataset_test"]
        self._end_test = config["config_model"]["end_dataset_test"]
        self._positive_intrusion_multiplier = config["config_model"]["positive_intrusion_multiplier"]
        self._positive_normal_multiplier = config["config_model"]["positive_normal_multiplier"]
        self._negative_intrusion_multiplier = config["config_model"]["negative_intrusion_multiplier"]
        self._negative_normal_multiplier = config["config_model"]["negative_normal_multiplier"]
        self._max_steps = config["config_model"].get("n_steps", 10000)
        self._n_episodes_train = config["config_model"]["n_episodes_train"]
        self._proportion_intrusion = config["config_model"]["proportion_intrusion"]
        self._is_generalization = config["config_model"]["generalization"]
        self._matrix_normal_points = []
        self._matrix_anomaly_points = []
        self._distance_features = []
        self._proportion_normal = 1 - self._proportion_intrusion
        self._dataset_type = dataset_type
        self._env_data, self._env_labels = self.__build_dataset(dataset)
        self._start_index = 0
        self._env_index = 0
        self._intrusion_counter = 0
        self._normal_counter = 0

    def __build_dataset(self, paths_dictionary: typing.Dict):
        
        features_array = np.load(paths_dictionary["X_path"])
        features_array = features_array.f.arr_0      
                
        if self._dataset_type == "train":
            if self._is_generalization:
                labels_array = pd.read_csv(paths_dictionary["y_path"])
                labels_array = np.array(labels_array["Class"].values)
                labels_array = labels_array.f.arr_0
                print(labels_array)
            else:
                labels_array = np.load(paths_dictionary["y_path"])
                labels_array = labels_array.f.arr_0
            
            features_array = features_array[self._start_train:self._end_train]
            labels_array = labels_array[self._start_train:self._end_train]
        
        else:
            if self._is_generalization:
                labels_array = pd.read_csv(paths_dictionary["y_path"])
                labels_array = np.array(labels_array["Class"].values)
                labels_array = labels_array.f.arr_0
                print(labels_array)
            else:
                labels_array = np.load(paths_dictionary["y_path"])
                labels_array = labels_array.f.arr_0
            
            features_array = features_array[self._start_test:self._end_test]
            labels_array = labels_array[self._start_test:self._end_test]
        
        print(f"Built dataset with shape: {features_array.shape}")
        print(f"Loaded labels as:{labels_array}")
        
        return features_array, labels_array
    
    def reset_env(self):
        max_start = len(self._env_data) - self._max_steps - 1
        self._env_index = 0

        # Randomized reset of enviroment state 
        self._start_index = random.randint(0, max_start)

        state = self._env_data[self._start_index]
        return torch.from_numpy(state).unsqueeze(dim=0).float()
    
    def reset_env_test(self):
        self._env_index = 0
        state = self._env_data[0]
        return torch.from_numpy(state).unsqueeze(dim=0).float()

    def step_env(self, action):
        action = action.item()
        true_idx = self._start_index + self._env_index
        reward = torch.tensor(self.__compute_reward(action, true_idx)).view(1, -1).float()
        
        self._env_index += 1
        
        done = 1 if self._env_index >= self._max_steps else 0

        next_idx = self._start_index + self._env_index
        next_state = torch.from_numpy(self._env_data[next_idx]).unsqueeze(dim=0).float()

        return next_state, reward, torch.tensor(done).view(1, -1)
    
    def step_env_test(self, action):
        action = action.item()
        reward = torch.tensor(self.__compute_reward(action, self._env_index)).view(1, -1).float()
        self._env_index += 1
        next_state = torch.from_numpy(self._env_data[self._env_index]).unsqueeze(dim=0).float()
        done = 1 if self._env_index == self._env_data.shape[0] - 1 else 0
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done  
        
    def __compute_reward(self, action, true_idx):
        true_label = self._env_labels[true_idx]
        if action == 1 and true_label == 1:
            return self._positive_reward * self._positive_intrusion_multiplier
        elif action == 0 and true_label == 0:
            return self._positive_reward * self._positive_normal_multiplier
        elif action == 0 and true_label == 1:
            return self._negative_reward * self._negative_intrusion_multiplier
        elif action == 1 and true_label == 0:
            return self._negative_reward * self._negative_normal_multiplier

class DQNModelGenerator():
    def __init__(self, config: typing.Dict, features_names):

        self._features_names = features_names
        self._state_size = len(features_names)
        self._action_size = NUM_ACTIONS
        self._alpha = config["config_model"]["alpha"]
        self._gamma = config["config_model"]["gamma"]
        self._target_update_frequency = config["config_model"]["target_update_frequency"]
        self._epsilon = config["config_model"]["epsilon"]
        self._epsilon_min = config["config_model"]["epsilon_min"]
        self._n_episodes_train = config["config_model"]["n_episodes_train"]
        self._n_episodes_test = config["config_model"]["n_episodes_test"]
        self._n_steps = config["config_model"]["n_steps"]
        self._checkpoint_frequency = config["config_model"]["checkpoint_frequency_per_episode"]
        self._replay_memory = ReplayMemory(config)
        self._environment_train = Environment(config, config["dataset_train_load_paths"], "train")
        self._environment_test = Environment(config, config["dataset_test_load_paths"], "test")
        self._logger = logging.getLogger(__name__)
        self._cur_episode = 0
        logging.basicConfig(
        filename=LOG_FILE_PATH,  
        filemode='a',         
        format='%(asctime)s - %(levelname)s - %(message)s', 
        level=logging.INFO)
        self.q_network = self.__build_network()
        self._checkpoint_path = config["config_model"]["save_path"]
        self._target_q_network = copy.deepcopy(self.q_network).eval()
        self._start_time = 0
        self._end_time = 0
        self._c_report = None
        self._confusion_matrix = None
    
    def __build_network(self):
        return nn.Sequential(
            nn.Linear(self._state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_ACTIONS)
        )
    
    def __policy(self, state):
        if torch.rand(1) < self._epsilon:
            return torch.randint(NUM_ACTIONS, (1,1))
        av = self.q_network(state).detach()
        return torch.argmax(av, dim = 1, keepdim=True)
    
    def __policy_test(self, state):
        av = self.q_network(state).detach()
        return torch.argmax(av, dim = 1, keepdim=True)
        
    def deep_q_learning(self):
        """ Initialize Neural Network Optimizer """

        optim = AdamW(self.q_network.parameters(), lr=self._alpha)
        stats = {'MSE Loss': [], 'Returns': []}

        for episode in tqdm(range(1, self._n_episodes_train + 1)):
            
            state = self._environment_train.reset_env()
            done = False
            ep_return = 0

            while not done:
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
                    target_b = reward_batch + (self._gamma * next_qsa_b * (1 - done_batch))

                    # Calculate the mean squared error (loss function)
                    loss = F.mse_loss(qsa_b, target_b)
                    self.q_network.zero_grad()
                    # Do the backpropagation 
                    loss.backward()
                    # Update the neural network weights and biases
                    optim.step()

                    loss_val = loss.item()
                    stats['MSE Loss'].append(loss_val)
                    wandb.log({
                        "MSE Loss": loss_val,
                        "Episode": episode,
                        "Step": episode * len(stats['MSE Loss']),
                        "Epsilon": self._epsilon
                    }, step=episode)

                state = next_state
                ep_return += reward.item()
                
            if self._epsilon > self._epsilon_min:
                self._epsilon *= 0.999

            stats["Returns"].append(ep_return)
            
            if episode % self._target_update_frequency == 0:
                self._target_q_network.load_state_dict(self.q_network.state_dict())
            
            if episode % self._checkpoint_frequency == 0:
                checkpoint_path = f"{self._checkpoint_path}_ep{episode}.pth"
                print(f"CHECKPOINT OF EPISODE {episode}")
                torch.save(self.q_network.state_dict(), checkpoint_path)
                self._cur_episode = episode
                self.test_model()
                wandb.save(checkpoint_path)
                
            # Wandb log after each episode
            wandb.log({
                "Episode Return": ep_return,
                "Episode": episode
            }, step=self._cur_episode)

        return stats
    
    def test_model(self):
        y_true = []
        y_pred = []

        state = self._environment_test.reset_env_test() 
        done = False
        
        self._start_time = time.time()

        while not done:
            action = self.__policy_test(state) 
            
            y_true.append(self._environment_test._env_labels[self._environment_test._env_index])  
            y_pred.append(action.item())
            next_state, reward, done = self._environment_test.step_env_test(action)
            state = next_state
            
        self._end_time = time.time()

        print("Classification Report:")
        self._c_report = classification_report(y_true, y_pred, target_names=["Intrusion", "Normal"], output_dict=True)
        print(self._c_report)
        
        print("\nConfusion Matrix:")
        self._confusion_matrix = confusion_matrix(y_true, y_pred)
        print(self._confusion_matrix)
        
        wandb.log({
            "Test Precision": self._c_report["Intrusion"]["precision"],
            "Test Recall": self._c_report["Intrusion"]["recall"],
            "Test F1": self._c_report["Intrusion"]["f1-score"],
            "Test Accuracy": self._c_report["accuracy"],
            "Test Time (s)": self._end_time - self._start_time
        }, step=self._cur_episode)

                
    def test_model_generalized(self):
        y_true = []
        y_pred = []
        
        # Adapt environment so that step test and reset env test make matrix for distance features calculations for each state
        
        
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