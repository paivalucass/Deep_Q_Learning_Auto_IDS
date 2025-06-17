import json
import argparse
import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
from torch import nn as nn
from scapy.all import *
from collections import deque
from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

DEFAULT_WINDOW_SIZE = 44
NUM_ACTIONS = 2
DEFAULT_NUMBER_OF_BYTES = 58
DEFAULT_WINDOW_SLIDE = 1
AVTP_PACKETS_LENGHT = 438
DEFAULT_LABELING_SCHEMA = "AVTP_Intrusion_dataset"
DEFAULT_DATASET = "AVTP_Intrusion"
DEFAULT_SUM_X = False
DEFAULT_RANDOMIZE = False

class DQN_IDS():
    def __init__(self, dqn):
        self.dqn = dqn
        
        self.ids()
        
    def ids(self):
        console = Console()
        
        console.print("\n[bold blue]==== Starting DQN-based Intrusion Detection System ====\n[/bold blue]")

        # Keep last 10 predictions as deque (timestamp, prediction)
        history = deque(maxlen=10)

        intrusion_count = 0
        normal_count = 0

        with Live(refresh_per_second=2, console=console, screen=False) as live:
            while True:
                packets = sniff(iface='veth0', count=44)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                converted_packets = self.__convert_packets(packets)
                selected_packets = self.__select_packets_bytes(converted_packets)
                diff_module_packets = self.__calculate_difference_module(selected_packets)
                nibbles_packets = self.__split_into_nibbles(diff_module_packets)
                aggregated_package = self.__entropy_aggregation(nibbles_packets)
                package = torch.from_numpy(aggregated_package).unsqueeze(dim=0).float()
                prediction = self.__policy_test(package)

                # Update counts
                if prediction == 1:
                    intrusion_count += 1
                else:
                    normal_count += 1

                # Add to history
                history.append((timestamp, prediction))

                # Build main status table
                status_table = Table(title="📡 DQN IDS - Real-time Status", expand=True)
                status_table.add_column("Metric", style="cyan", no_wrap=True)
                status_table.add_column("Value", style="magenta")

                status_table.add_row("Captured Packets", str(len(packets)))
                status_table.add_row("Intrusions Detected", str(intrusion_count))
                status_table.add_row("Normal Traffic", str(normal_count))

                # Build last 10 predictions table
                history_table = Table(title="🕒 Last 10 Detections", expand=True)
                history_table.add_column("Timestamp", style="green")
                history_table.add_column("Prediction", style="bold")

                for ts, pred in reversed(history):
                    pred_str = "[red]INTRUSION[/red]" if pred == 1 else "[green]NORMAL[/green]"
                    history_table.add_row(ts, pred_str)

                # Prepare packet snapshot info
                pkt_info_lines = []
                for idx, pkt in enumerate(packets[:3]):
                    raw_bytes = raw(pkt)[:16]
                    hex_dump = ' '.join(f"{b:02x}" for b in raw_bytes)
                    pkt_info_lines.append(f"[{idx+1}] Len={len(pkt)} | Head: {hex_dump}")

                pkt_info_text = Text("\n".join(pkt_info_lines), style="yellow")
                packet_panel = Panel(pkt_info_text, title="Packet Snapshots (First 3)", border_style="bright_blue")

                # Compose full layout group
                group = Group(status_table, history_table, packet_panel)

                # Update live output
                live.update(group)

    def __policy_test(self, state):
        av = self.dqn(state).detach()
        action = torch.argmax(av, dim = 1, keepdim=True)
        return action.item()
            
    def __convert_packets(self, packets):
        converted_packets_list = []
        for packet in packets:
            converted_packet = np.frombuffer(raw(packet), dtype='uint8')
            converted_packet_len = len(converted_packet)
            if converted_packet_len < DEFAULT_NUMBER_OF_BYTES:
                bytes_to_pad = DEFAULT_NUMBER_OF_BYTES - converted_packet_len
                converted_packet = np.pad(converted_packet, (0, bytes_to_pad), 'constant')
            else:
                converted_packet = converted_packet[0:DEFAULT_NUMBER_OF_BYTES]

            converted_packets_list.append(converted_packet)
        
        return np.array(converted_packets_list, dtype='uint8')
            
    def __entropy_aggregation(self, arr):
        # Calculate entropy along the axis=0 for each column independently
        return np.apply_along_axis(lambda col: entropy(np.histogram(col, bins=10, density=True)[0]), axis=0, arr=arr)    
        
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
        
    def __select_packets_bytes(self, packets_list):
        selected_packets = packets_list[:, 0:DEFAULT_NUMBER_OF_BYTES]

        return np.array(selected_packets, dtype='uint8')
    
    def __calculate_difference_module(self, selected_packets):
        difference_array = np.diff(selected_packets, axis=0)
        difference_module = np.mod(difference_array, 256)

        return difference_module
        
        
def main():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--config', required=True, help='JSON File containing the configs for the specified feature generation method')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as config_file:
            config = json.load(config_file)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")
        return
    
    features_names = [f"feat_{i}" for i in range(config["config_model"]["feature_size"])]
        
    
    model_path = "/home/lucas/DQL_IDS/Deep_Q_Learning_Auto_IDS/trained_models/DQN_equal_rewards_all_labels_1-2_positive_new_balancing_400_steps_ep7000.pth"
    print(f"Loading model from: {model_path}")
    q_network = build_network()
    q_network.load_state_dict(torch.load(model_path))
    DQN_IDS(dqn=q_network)
    
def build_network():
    return nn.Sequential(
        nn.Linear(116, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, NUM_ACTIONS)
    )

if __name__ == "__main__":
    main()
