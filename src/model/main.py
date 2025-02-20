import model_generator
import time
import json
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--config', required=True, help='JSON File containing the configs for the specified feature generation method')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as config:
            config = json.load(config)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")
        
    # Create a features name array
    # features_names = np.array([f"feat_{i}" for i in range(config["config_model"]["feature_size"])]) # Alternative way
    
    features_names = []
    for i in range(config["config_model"]["feature_size"]):
        features_names.append(f"feat_{i}")
        
    # Load model    
    model = model_generator.DQLModelGenerator(config, features_names)
    
    feature_generator_name = config['feature_generator']
    feature_generator_config = config['config']
    feature_generator_paths = config['paths']
    feature_generator_load_paths = config['load_paths']
    mode = config["mode"]
    
    trained_model_saving_folder = config["config_model"]["pkl_saving_path"]
    trained_model_saving_path = f"{trained_model_saving_folder}/TRAINED_{feature_generator_config['labeling_schema']}_AggregationMethod_{feature_generator_config['aggregation_method']}_Wsize_{feature_generator_config['window_size']}_Cols_{feature_generator_config['number_of_bytes'] * 2}_Wslide_{feature_generator_config['window_slide']}_MC_{feature_generator_config['multiclass']}_Algorithm_{config['config_model']['algorithm']}_RemovedAttack_{feature_generator_config['remove_attack']}.pkl"

if __name__ == "__main__":
    main()