import pandas as pd
import numpy as np
import time
import json
import argparse 
from scapy.all import *
import generator 

# AVAILABLE_MODEL_GENERATORS

AVAILABLE_FEATURE_GENERATORS = {
    "Deep_Q_Learning": generator.DQL_Generator
}

def main():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--config', required=True, help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--bench_time', action='store_true', help='Flag to execute the feature generator execution time benchmark')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as config:
            config_dict = json.load(config)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")

    print("##### Loaded paths dictionary #####")
    print(json.dumps(config_dict, indent=4, sort_keys=True))

    feature_generator_name = config_dict['feature_generator']
    feature_generator_config = config_dict['config']
    feature_generator_paths = config_dict['paths']
    feature_generator_load_paths = config_dict['load_paths']
    mode = config_dict["mode"]

    if feature_generator_name not in AVAILABLE_FEATURE_GENERATORS:
        raise KeyError(f"Selected feature generator: {feature_generator_name} is NOT available!")

    selected_feature_generator = AVAILABLE_FEATURE_GENERATORS[feature_generator_name](feature_generator_config)
    print(f"> Selected feature generator: {feature_generator_name}")

    if mode == "aggregate_features":
        
        print(f"> Selected aggregation method: {feature_generator_config['aggregation_method']}")
        if feature_generator_config["aggregation_method"] == "Entropy":
            features, labels = selected_feature_generator.load_features_entropy(feature_generator_load_paths)
        elif feature_generator_config["aggregation_method"] == None:
            features, labels = selected_feature_generator.load_features_no_aggregation(feature_generator_load_paths)
            
        np.savez(f"{feature_generator_load_paths['output_path']}/X_{feature_generator_config['suffix']}_{feature_generator_config['labeling_schema']}_AggregationMethod_{feature_generator_config['aggregation_method']}_Wsize_{feature_generator_config['window_size']}_Cols_{feature_generator_config['number_of_bytes'] * 2}_Wslide_{feature_generator_config['window_slide']}_MC_{feature_generator_config['multiclass']}_RemovedAttack_{feature_generator_config['remove_attack']}", features)
        np.savez(f"{feature_generator_load_paths['output_path']}/Y_{feature_generator_config['suffix']}_{feature_generator_config['labeling_schema']}_AggregationMethod_{feature_generator_config['aggregation_method']}_Wsize_{feature_generator_config['window_size']}_Cols_{feature_generator_config['number_of_bytes'] * 2}_Wslide_{feature_generator_config['window_slide']}_MC_{feature_generator_config['multiclass']}_RemovedAttack_{feature_generator_config['remove_attack']}", labels)

    elif mode == "generate_features":
        
        if args.bench_time:
            print("> Execution time benchmark generation")
            selected_feature_generator.benchmark_execution_time()
        else:
            print("> Generating features...")
            selected_feature_generator.generate_features(feature_generator_paths)

    print("Feature generator successfully executed!")

if __name__ == "__main__":
    main()
