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
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import wandb
import pandas as pd
from features import DQNFeatureGenerator
from collections import deque
from scipy.spatial.distance import cdist

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
    
class Buffer():
    def __init__(self, x_data, config: typing.Dict):
        print(">> Assembling Isolation Forest...")
        self._debug = config["debug_mode"]
        iso_forest = IsolationForest(random_state=42, contamination='auto')
        iso_forest.fit(x_data)
        self._normal_buffer_size = config["config_model"]["normal_buffer_size"]
        self._anonaly_buffer_size = config["config_model"]["anomaly_buffer_size"]
        anomaly_scores = -iso_forest.score_samples(x_data)
        scaler = MinMaxScaler()
        self._anomaly_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
        self._normal_buffer = deque(maxlen=self._normal_buffer_size)
        self._anomaly_buffer = deque(maxlen=self._anonaly_buffer_size)
        self._k_neighbor = config["config_model"]["k-neighbor"]
        
    def update_buffer(self, packet, label):
        if label == 0:
            self._normal_buffer.append(packet)
        elif label == 1:
            self._anomaly_buffer.append(packet)
            
    def _minmax(self, feature, minimum, maximum):
        return (feature - minimum) / (maximum - minimum)
    
    def extract_state(self, packet, index):
        anomaly_score = self._anomaly_scores[index]
        
        if self._debug:
            print(f"anomaly score: {anomaly_score}")
            print(f"len normal buffer: {self._normal_buffer}")
            print(f"len anomaly buffer: {self._anomaly_buffer}")

        if len(self._normal_buffer) > 0:
            normal_points = np.array(self._normal_buffer)
            dist_normals = cdist([packet], normal_points, metric='euclidean')[0]
            avg_dist_normal = np.mean(dist_normals)
            min_dist_normal = np.min(dist_normals)
        else:
            avg_dist_normal = 0
            min_dist_normal = 0

        if len(self._anomaly_buffer) > 0:
            anomaly_points = np.array(self._anomaly_buffer)
            dist_anomalies = cdist([packet], anomaly_points, metric='euclidean')[0]
            avg_dist_anomaly = np.mean(dist_anomalies)
            min_dist_anomaly = np.min(dist_anomalies)
        else:
            avg_dist_anomaly = 0
            min_dist_anomaly = 0

        combined = np.array(list(self._normal_buffer) + list(self._anomaly_buffer))
        labels = np.array([0]*len(self._normal_buffer) + [1]*len(self._anomaly_buffer))

        if len(combined) > 0:
            distances = cdist([packet], combined, metric='euclidean')[0]
            k = min(self._k_neighbor, len(distances))
            knn_indices = np.argsort(distances)[:k]
            knn_labels = labels[knn_indices]
            neighborhood = 1 if np.any(knn_labels == 1) else 0
        else:
            neighborhood = 0
        
        if self._debug:
            print(f"min_dist_normal: {min_dist_normal}")
            print(f"avg_dist_normal: {avg_dist_normal}")
            print(f"min_dist_anomaly: {min_dist_anomaly}")
            print(f"avg_dist_anomaly: {avg_dist_anomaly}")
        
        # Normalization
        min_dist_normal = self._minmax(min_dist_normal, 0, 161.554944214)
        avg_dist_normal = self._minmax(avg_dist_normal, 0, 161.554944214)
        min_dist_anomaly = self._minmax(min_dist_anomaly, 0, 161.554944214)
        avg_dist_anomaly = self._minmax(avg_dist_anomaly, 0, 161.554944214)

        return np.array([
            anomaly_score,
            min_dist_normal,
            min_dist_anomaly,
            avg_dist_normal,
            avg_dist_anomaly,
            neighborhood
        ], dtype=np.float32)
        
class Environment():
    def __init__(self, config: typing.Dict, dataset_type = "train"):
        self._env_index = 0
        self._debug = config["debug_mode"]
        self._config = config
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
        self._feature_generator = DQNFeatureGenerator(config, dataset_type=dataset_type)
        self._env_data, self._env_labels, self._env_labels_multiclass = self.__build_dataset()
        self.buffer = Buffer(x_data=self._env_data, config=config)
        self._start_index = 0
        self._env_index = 0
        self._intrusion_counter = 0
        self._normal_counter = 0

    def __build_dataset(self):
        return self._feature_generator.generate_features()
    
    def reset_env(self):
        max_start = len(self._env_data) - self._max_steps - 1
        self._env_index = 0

        # Randomized reset of enviroment state 
        self._start_index = random.randint(0, max_start)

        raw_packet = self._env_data[self._start_index]
        state = self.buffer.extract_state(raw_packet, self._start_index)
        
        if self._debug:
            print(f"STATE: {state}")
        
        return torch.from_numpy(state).unsqueeze(dim=0).float(), raw_packet
    
    def reset_env_test(self):
        self._env_index = 0
        
        raw_packet = self._env_data[0]
        state = self.buffer.extract_state(raw_packet, self._start_index)
        return torch.from_numpy(state).unsqueeze(dim=0).float(), raw_packet

    def step_env(self, action):
        action = action.item()
        true_idx = self._start_index + self._env_index
        reward = torch.tensor(self.__compute_reward(action, true_idx)).view(1, -1).float()
        
        self._env_index += 1
        
        done = 1 if self._env_index >= self._max_steps else 0

        next_idx = self._start_index + self._env_index        
        
        raw_packet = self._env_data[next_idx]
        next_state = self.buffer.extract_state(raw_packet, next_idx)
        
        if self._debug:
            print(f"STATE: {next_state}")
            
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        

        return next_state, reward, torch.tensor(done).view(1, -1), raw_packet
    
    def step_env_test(self, action):
        action = action.item()
        reward = torch.tensor(self.__compute_reward(action, self._env_index)).view(1, -1).float()
        self._env_index += 1
        raw_packet = self._env_data[self._env_index]
        next_state = self.buffer.extract_state(raw_packet, self._env_index)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        done = 1 if self._env_index == self._env_data.shape[0] - 1 else 0
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done, raw_packet
        
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
        self._debug = config["debug_mode"]
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
        self._environment_train = Environment(config, "train")
        self._environment_test = Environment(config, "test")
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
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, NUM_ACTIONS)
        )
    
    def __policy(self, state):
        self.q_network.train()
        if torch.rand(1) < self._epsilon:
            return torch.randint(NUM_ACTIONS, (1,1))
        av = self.q_network(state).detach()
        return torch.argmax(av, dim = 1, keepdim=True)
    
    def __policy_test(self, state):
        self.q_network.eval()
        av = self.q_network(state).detach()
        return torch.argmax(av, dim = 1, keepdim=True)
        
    def deep_q_learning(self):
        """ Initialize Neural Network Optimizer """
        self.q_network.train()

        optim = AdamW(self.q_network.parameters(), lr=self._alpha)
        stats = {'MSE Loss': [], 'Returns': []}

        for episode in tqdm(range(1, self._n_episodes_train + 1)):
            
            state, raw_packet = self._environment_train.reset_env()
            done = False
            ep_return = 0

            while not done:
                # Calculate an action to be taken, based on the policy output
                action = self.__policy(state)
                self._environment_train.buffer.update_buffer(raw_packet, action)
                
                # Apply the action at the enviroment and returns new state and reward 
                next_state, reward, done, raw_packet = self._environment_train.step_env(action)
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
                        "Epsilon": self._epsilon
                    }, step=episode)

                state = next_state
                ep_return += reward.item()
                
            if self._epsilon > self._epsilon_min:
                self._epsilon *= 0.9998

            stats["Returns"].append(ep_return)
            
            if episode % self._target_update_frequency == 0:
                self._target_q_network.load_state_dict(self.q_network.state_dict())
                self._target_q_network.eval()
            
            if episode % self._checkpoint_frequency == 0:
                checkpoint_path = f"{self._checkpoint_path}_ep{episode}.pth"
                print(f"CHECKPOINT OF EPISODE {episode}")
                torch.save(self.q_network.state_dict(), checkpoint_path)
                self._cur_episode = episode
                self.test_model()
                wandb.save(checkpoint_path)
                self.q_network.train()
                
            # Wandb log after each episode
            wandb.log({
                "Episode Return": ep_return,
                "Episode": episode
            }, step=episode)

        return stats
    
    def test_model(self):
        self.q_network.eval()
        y_true = []
        y_pred = []
        test_reward = []

        state, raw_packet = self._environment_test.reset_env_test() 
        done = False
        
        self._start_time = time.time()

        while not done:
            action = self.__policy_test(state) 
            self._environment_test.buffer.update_buffer(raw_packet, action)
            
            y_true.append(self._environment_test._env_labels[self._environment_test._env_index])  
            y_pred.append(action.item())
            next_state, reward, done, raw_packet = self._environment_test.step_env_test(action)
            test_reward.append(reward)
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
            "Test Reward": test_reward,
            "Test Accuracy": self._c_report["accuracy"],
            "Test Time (s)": self._end_time - self._start_time
        }, step=self._cur_episode)
