import model_generator
import time
import json
import argparse
import pickle
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--config', required=True, help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--mode', required=True, help='Train or Test mode')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as config:
            config = json.load(config)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")
    
    features_names = []
    for i in range(config["config_model"]["feature_size"]):
        features_names.append(f"feat_{i}")
        
    if args.mode == "Train":
        # Load model    
        dql = model_generator.DQLModelGenerator(config, features_names)
        
        stats = dql._deep_q_learning()
        q_network = dql.q_network
        
        with open("/home/slurm/pesgradivn/lcap/Deep_Q_Learning_Auto_IDS/trained_models/dql_all_labels.pkl", "wb") as file:
            pickle.dump(q_network, file)
            
        with open("/home/slurm/pesgradivn/lcap/Deep_Q_Learning_Auto_IDS/trained_models/stats_all_labels.pkl", "wb") as file:
            pickle.dump(stats, file)
                    
        dql.test_model()
    else:
        dql = model_generator.DQLModelGenerator(config, features_names)
        stats = None
        
        with open("/home/slurm/pesgradivn/lcap/Deep_Q_Learning_Auto_IDS/trained_models/dql_2.pkl", "rb") as file:
            dql.q_network = pickle.load(file)
        
        dql.test_model()
        
        dql.save_metric(config)
        
if __name__ == "__main__":
    main()