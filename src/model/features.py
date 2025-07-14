import pandas as pd
import numpy as np
import typing
import time
import json
from scapy.all import *
from tqdm import tqdm

DEFAULT_WINDOW_SIZE = 44
DEFAULT_NUMBER_OF_BYTES = 58
DEFAULT_WINDOW_SLIDE = 1
AVTP_PACKETS_LENGHT = 438
DEFAULT_LABELING_SCHEMA = "AVTP_Intrusion_dataset"
DEFAULT_DATASET = "TOW_IDS_dataset"
DEFAULT_SUM_X = False
DEFAULT_RANDOMIZE = False

class DQNFeatureGenerator():
    def __init__(self, config: typing.Dict, dataset_type="train"):
        self._window_size = config["config"].get('window_size', DEFAULT_WINDOW_SIZE)
        self._number_of_bytes = config["config"].get('number_of_bytes', DEFAULT_NUMBER_OF_BYTES)
        self._window_slide = config["config"].get('window_slide', DEFAULT_WINDOW_SLIDE)
        self._number_of_columns = self._number_of_bytes * 2
        self._labeling_schema = config["config"].get('labeling_schema', DEFAULT_LABELING_SCHEMA)
        self._sum_x = config["config"].get('sum_x', DEFAULT_SUM_X)
        self._data_suffix = config["config"].get('suffix')
        self._reduced_dataset = config["config"].get('reduced_dataset')
        self._dataset = config["config"].get('dataset', DEFAULT_DATASET)
        self._filter_avtp_packets = True if self._dataset == "AVTP_Intrusion_dataset" else False
        self._paths_dictionary = config["paths"]
        self._dataset_type = dataset_type
        
    def generate_features(self):
        AVAILABLE_DATASETS = {
            "AVTP_Intrusion_dataset": self.__avtp_dataset_generate_features,
            "TOW_IDS_dataset": self.__tow_ids_dataset_generate_features
        }

        if self._dataset not in AVAILABLE_DATASETS:
            raise KeyError(f"Selected dataset: {self._dataset} is NOT available for DQN IDS Feature Generator!")

        return AVAILABLE_DATASETS[self._dataset]()
        
    def __tow_ids_dataset_generate_features(self):
        print("USING TOW-IDS")
        # Load raw packets
        if self._dataset_type == "train":
            labels = pd.read_csv(self._paths_dictionary["y_train_path"], header=None, names=["index", "Class", "Description"])
            raw_packets = rdpcap(self._paths_dictionary["training_packets_path"])
        elif self._dataset_type == "test":
            labels = pd.read_csv(self._paths_dictionary["y_test_path"], header=None, names=["index", "Class", "Description"])
            raw_packets = rdpcap(self._paths_dictionary["test_packets_path"])
            
        CLASS_MAP = {
            "normal": 0,
            "c_r": 1,
            "c_d": 2,
            "m_f": 3,
            "p_i": 4,
            "f_i": 5
        }
            
        labels_multiclass = labels.drop(columns=["index", "Class"])
        labels_multiclass = labels_multiclass["Description"].str.lower().map(CLASS_MAP).to_numpy()
        labels_binary = labels.drop(columns=["index", "Description"])
        labels_binary = labels_binary["Class"].apply(lambda x: 0 if x.lower() == "normal" else 1).to_numpy()

        print(">> Converting raw packets...")
        converted_packets = self.__convert_packages(raw_packets)

        print(">> Preprocessing raw packets...")
        preprocessed_packets = self.__preprocess_raw_packets(converted_packets, split_into_nibbles=True)
        
        print(f"packets: {preprocessed_packets[:50]}")
        print(f"labels binary: {labels_binary[:50]}")
        print(f"labels multiclass: {labels_multiclass[:50]}")
        
        return preprocessed_packets, labels_binary, labels_multiclass
    
    def __avtp_dataset_generate_features(self):
        print("USING AVTP DATASET")
        # Load Packets
        X = np.load(f"{self._paths_dictionary['avtp_test_path']}.npz")
        Y = pd.read_csv(f"{self._paths_dictionary['avtp_test_path']}.csv", header=None, names=["index", "Class"])
        labels_binary = Y.drop(columns=["index"])
        labels_binary = labels_binary["Class"].apply(lambda x: 0 if x == 0 else 1).to_numpy()
        X = X.f.arr_0
        # Load 
            
        print(f"packets: {X[:50]}")
        print(f"size packets: {len(X[0])}")
        print(f"labels binary: {labels_binary[:50]}")
        
        return X, labels_binary, None
    
    def avtp_dataset_process(self):
        print("GENERATING AVTP DATASET")
        # Load raw packets
        print(f"PATH ATTACK ONLY: {self._paths_dictionary['injected_only_path']}")
        #Injected Only
        raw_injected_only_packets = self.__read_raw_packets(self._paths_dictionary['injected_only_path'])
        print(f"size only attacks: {len(raw_injected_only_packets)}")
        injected_only_packets_array = self.__convert_packages(raw_injected_only_packets)
        
        count = 1
        for dataset in self._paths_dictionary['avtp_dataset_path']:
            count += 1
            print(f"PATH DATASET: {dataset}")
            raw_dataset_packets = self.__read_raw_packets(dataset)
            print(f"size all dataset: {len(raw_dataset_packets)}")

            # Convert packets
            packets_array = self.__convert_packages(raw_dataset_packets)

            # Preprocess packets
            preprocessed_packets = self.__preprocess_raw_packets(packets_array, split_into_nibbles=True)

            # Generate labels
            labels_binary = self.__generate_labels(packets_array, injected_only_packets_array)
                
            print(f"packets: {preprocessed_packets[:50]}")
            print(f"size packets: {len(preprocessed_packets[0])}")
            print(f"number of packets: {len(preprocessed_packets)}")
            print(f"labels binary: {labels_binary[:50]}")
            print(f"number of labels: {len(labels_binary)}")
            
            np.savez(f"{self._paths_dictionary['avtp_output_path']}/avtp_{count}", preprocessed_packets)
            
            y_df = pd.DataFrame(labels_binary, columns=["Class"])
            y_df.to_csv(f"{self._paths_dictionary['avtp_output_path']}/avtp_{count}.csv")
    
    def __generate_labels(self, packets_list, injected_packets):
        labels_list = []

        for packet in packets_list:
            current_label = 0

            if self.__is_array_in_list_of_arrays(packet, injected_packets):
                current_label = 1

            labels_list.append(current_label)

        return labels_list
    
    def __is_array_in_list_of_arrays(self, array_to_check, list_np_arrays):
        # Reference:
        # https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        is_in_list = np.any(np.all(array_to_check == list_np_arrays, axis=1))

        return is_in_list
    
    def __read_raw_packets(self, pcap_filepath):
        print(pcap_filepath)
        raw_packets = rdpcap(pcap_filepath)

        raw_packets_list = []

        for packet in raw_packets:
            if self._filter_avtp_packets:
                if (len(packet) == AVTP_PACKETS_LENGHT):
                    raw_packets_list.append(raw(packet))
            else:
                raw_packets_list.append(raw(packet))

        return raw_packets_list
            
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
    
if __name__ ==  "__main__":
    config_file = "/home/slurm/pesgradivn/lcap/Deep_Q_Learning_Auto_IDS/jsons/dql.json"
    with open(config_file, 'r') as config_file:
            config = json.load(config_file)
    generator = DQNFeatureGenerator(config)
    generator.avtp_dataset_process()
