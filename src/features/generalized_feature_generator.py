import pandas as pd
import numpy as np
import typing
import time
from scipy.stats import entropy
from scapy.all import *
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import joblib

import abstract_feature_generator
import labeling_schemas

from sklearn.preprocessing import OneHotEncoder

DEFAULT_WINDOW_SIZE = 44
DEFAULT_NUMBER_OF_BYTES = 58
DEFAULT_WINDOW_SLIDE = 1
AVTP_PACKETS_LENGHT = 438
DEFAULT_LABELING_SCHEMA = "AVTP_Intrusion_dataset"
DEFAULT_DATASET = "AVTP_Intrusion"
DEFAULT_SUM_X = False
DEFAULT_RANDOMIZE = False

LABELING_SCHEMA_FACTORY = {
    "AVTP_Intrusion_dataset": labeling_schemas.avtp_intrusion_labeling_schema,
    "TOW_IDS_dataset_one_class": labeling_schemas.tow_ids_one_class_labeling_schema,
    "TOW_IDS_dataset_multi_class": labeling_schemas.tow_ids_multi_class_labeling_schema
}

class DQN_Generator(abstract_feature_generator.AbstractFeatureGenerator):
    def __init__(self, config: typing.Dict):
        self._window_size = config.get('window_size', DEFAULT_WINDOW_SIZE)
        self._number_of_bytes = config.get('number_of_bytes', DEFAULT_NUMBER_OF_BYTES)
        self._window_slide = config.get('window_slide', DEFAULT_WINDOW_SLIDE)
        self._number_of_columns = self._number_of_bytes * 2
        self._labeling_schema = config.get('labeling_schema', DEFAULT_LABELING_SCHEMA)
        self._sum_x = config.get('sum_x', DEFAULT_SUM_X)
        self._data_suffix = config.get('suffix')
        self._randomize = config.get('randomize', DEFAULT_RANDOMIZE)
        self._reduced_dataset = config.get('reduced_dataset')
        
        self._dataset = config.get('dataset', DEFAULT_DATASET)

        self._output_path_suffix = f"{self._labeling_schema}_Wsize_{self._window_size}_Cols_{self._number_of_columns}_Wslide_{self._window_slide}_MC_{self._multiclass}_sumX_{self._sum_x}_removedAttacks_{self._remove_attacks_list}"

        self._filter_avtp_packets = True if (self._labeling_schema) == "AVTP_Intrusion_dataset" else False
        print(f"filter_avtp_packets = {self._filter_avtp_packets}")

    def generate_features(self, paths_dictionary: typing.Dict):
        AVAILABLE_DATASETS = {
            "AVTP_Intrusion_dataset": self.__avtp_dataset_generate_features,
            "TOW_IDS_dataset": self.__tow_ids_dataset_generate_features
        }

        if self._dataset not in AVAILABLE_DATASETS:
            raise KeyError(f"Selected dataset: {self._dataset} is NOT available for DQN IDS Feature Generator!")

        feature_generator = AVAILABLE_DATASETS[self._dataset](paths_dictionary)

    def __tow_ids_dataset_generate_features(self, paths_dictionary: typing.Dict):
        # Load raw packets
        if self._data_suffix == "train":
            labels = pd.read_csv(paths_dictionary["y_train_path"], header=None, names=["index", "Class", "Description"])
            raw_packets = rdpcap(paths_dictionary["training_packets_path"])
        elif self._data_suffix == "test":
            labels = pd.read_csv(paths_dictionary["y_test_path"], header=None, names=["index", "Class", "Description"])
            raw_packets = rdpcap(paths_dictionary["test_packets_path"])
            
        labels = labels.drop(columns=["index", "Description"])
        labels = labels["Class"].apply(lambda x: 0 if x.lower() == "normal" else 1).to_numpy()

        print(">> Converting raw packets...")
        converted_packets = self.__convert_packages(raw_packets)

        # Preprocess packets
        print(">> Preprocessing raw packets...")
        preprocessed_packets = self.__preprocess_raw_packets(converted_packets, split_into_nibbles=False)

        print(f"len_preprocessed_packets = {len(preprocessed_packets)}")
        print(f"preprocessed_packets[0] = {preprocessed_packets[0]}")

        # Aggregate features and labels
        print(">> Extracting Features...")
        features_array, labels, scaler = self.__anomaly_score_and_distance(preprocessed_packets, labels)

        np.savez(f"{paths_dictionary['output_path']}/X_{self._data_suffix}_{self._output_path_suffix}", features_array)

        y_df = pd.DataFrame(labels, columns=["Class"])
        y_df.to_csv(f"{paths_dictionary['output_path']}/y_{self._data_suffix}_{self._output_path_suffix}.csv")
        
        if self._data_suffix == "train":
            joblib.dump(scaler, f"{paths_dictionary['output_path']}/scaler_{self._data_suffix}.pkl")
        
    def __convert_packages(self, raw_packets):
        
        converted_packets_list = []

        for raw_packet in raw_packets:
            converted_packet = np.frombuffer(raw(raw_packet), dtype='uint8')

            converted_packet_len = len(converted_packet)
            if converted_packet_len < self._number_of_bytes:
                bytes_to_pad = self._number_of_bytes - converted_packet_len
                converted_packet = np.pad(converted_packet, (0, bytes_to_pad), 'constant')
            else:
                converted_packet = converted_packet[0:self._number_of_bytes]

            converted_packets_list.append(converted_packet)

        return np.array(converted_packets_list, dtype='uint8')

    def __select_packets_bytes(self, packets_list):
        selected_packets = packets_list[:, 0:self._number_of_bytes]

        return np.array(selected_packets, dtype='uint8')

    def __split_into_nibbles(self, x1):
        # Ensure the dtype is large enough to hold the shifted values without overflow
        x1_np = x1.astype(np.uint8)

        # Prepare a mask to isolate nibbles. 0xF is 1111 in binary, which isolates a nibble.
        mask = 0xF

        # Extract nibbles.
        # The idea is to shift the original numbers right by 0 and 4 bits
        # and then mask off the lower 4 bits.
        nibbles = np.zeros((x1_np.shape[0], x1_np.shape[1] * 2), dtype=np.uint8)

        for i in range(0, 2):
            nibbles[:, int(not i)::2] = (x1_np >> (i * 4)) & mask

        return nibbles

    def __preprocess_raw_packets(self, converted_packets, split_into_nibbles=False):
        # Select first 58 bytes
        selected_packets = self.__select_packets_bytes(converted_packets)

        # Split difference into two nibbles
        if split_into_nibbles:
            # diff_module_packets = self.__create_nibbles_matrix(diff_module_packets)
            selected_packets = self.__split_into_nibbles(selected_packets)

        return selected_packets
    
    def __anomaly_score_and_distance_based_on_window_size(self, x_data, y_data):
        window_size = self._window_size
        n_samples = x_data.shape[0]
        
        iso_forest = IsolationForest(random_state=42, contamination='auto')
        iso_forest.fit(x_data)

        anomaly_scores = -iso_forest.score_samples(x_data)  # Negative to make higher = more anomalous

        distance_features = []
        labels = []

        for i in tqdm(range(n_samples)):
            # Define the window before the sample
            start_ix = max(0, i - window_size)
            end_ix = i  

            window_X = x_data[start_ix:end_ix]
            window_y = y_data[start_ix:end_ix]

            # Current point
            current_sample = x_data[i]

            # Split window points into normal and anomalous
            normal_points = window_X[window_y == 0]
            anomaly_points = window_X[window_y == 1]

            # Distances to normal and anomaly points
            distances_to_normals = cdist([current_sample], normal_points, metric='euclidean')[0] if len(normal_points) > 0 else []
            distances_to_anomalies = cdist([current_sample], anomaly_points, metric='euclidean')[0] if len(anomaly_points) > 0 else []

            # Compute averages (if any)
            avg_distance_normal = np.mean(distances_to_normals) if len(distances_to_normals) > 0 else 0
            avg_distance_anomaly = np.mean(distances_to_anomalies) if len(distances_to_anomalies) > 0 else 0

            # Compute minimum distance to an anomaly
            min_prev_dist_normal = np.min(distances_to_normals) if len(distances_to_normals) > 0 else 0
            min_prev_dist_anomaly = np.min(distances_to_anomalies) if len(distances_to_anomalies) > 0 else 0
            
            # Verify neighbors for anomalies 
            neighborhood = 1 if len(anomaly_points) > 0 else 0

            # Final feature vector for this point
            feature_vector = [
                anomaly_scores[i],
                min_prev_dist_normal,
                min_prev_dist_anomaly,
                avg_distance_normal,
                avg_distance_anomaly,
                neighborhood
            ]

            distance_features.append(feature_vector)
            labels.append(y_data[i])
            
        features_array = np.array(distance_features, dtype=np.float32)

        scaler = MinMaxScaler()
        features_array = scaler.fit_transform(features_array)

        return features_array, np.array(labels), scaler
    
    def __anomaly_score_and_distance(self, x_data, y_data):
        window_size = self._window_size
        n_samples = x_data.shape[0]
        
        iso_forest = IsolationForest(random_state=42, contamination='auto')
        iso_forest.fit(x_data)

        anomaly_scores = -iso_forest.score_samples(x_data)  # Negative to make higher = more anomalous
        
        distance_features = []
        labels = []

        if self._data_suffix == "train":
            
            matrix_normal_points = x_data[y_data == 0]
            matrix_anomaly_points = x_data[y_data == 1]

            for i in tqdm(range(n_samples)):
                # Define the window before the sample
                start_ix = max(0, i - window_size)
                end_ix = i  

                window_X = x_data[start_ix:end_ix]
                window_y = y_data[start_ix:end_ix]

                # Current point
                current_sample = x_data[i]

                # Split window points into normal and anomalous
                anomaly_points = window_X[window_y == 1]

                # Distances to normal and anomaly points
                distances_to_normals = cdist([current_sample], matrix_normal_points, metric='euclidean')[0] if len(matrix_normal_points) > 0 else []
                distances_to_anomalies = cdist([current_sample], matrix_anomaly_points, metric='euclidean')[0] if len(matrix_anomaly_points) > 0 else []

                # Compute averages (if any)
                avg_distance_normal = np.mean(distances_to_normals) if len(distances_to_normals) > 0 else 0
                avg_distance_anomaly = np.mean(distances_to_anomalies) if len(distances_to_anomalies) > 0 else 0

                # Compute minimum distance to an anomaly
                min_prev_dist_normal = np.min(distances_to_normals) if len(distances_to_normals) > 0 else 0
                min_prev_dist_anomaly = np.min(distances_to_anomalies) if len(distances_to_anomalies) > 0 else 0
                
                # Verify neighbors for anomalies
                neighborhood = 1 if len(anomaly_points) > 0 else 0

                # Final feature vector for this point
                feature_vector = [
                    anomaly_scores[i],
                    min_prev_dist_normal,
                    min_prev_dist_anomaly,
                    avg_distance_normal,
                    avg_distance_anomaly,
                    neighborhood
                ]

                distance_features.append(feature_vector)
                labels.append(y_data[i])

            features_array = np.array(distance_features, dtype=np.float32)

            scaler = MinMaxScaler()
            features_array = scaler.fit_transform(features_array)

            print(features_array[0:5000])
            
            return features_array, np.array(labels), scaler
        
        else:
            
            for i in tqdm(range(n_samples)):
                
                feature_vector = [
                    anomaly_scores[i],
                    0,
                    0,
                    0,
                    0,
                    0
                ]

                distance_features.append(feature_vector)
                labels.append(y_data[i])
                
            features_array = np.array(distance_features, dtype=np.float32)
            return features_array, np.array(labels), None