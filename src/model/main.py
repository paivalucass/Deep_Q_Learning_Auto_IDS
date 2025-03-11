import model_generator
import time
import json
import argparse
import pickle
import matplotlib.pyplot as plt


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
    
    features_names = []
    for i in range(config["config_model"]["feature_size"]):
        features_names.append(f"feat_{i}")
        
    # Load model    
    dql = model_generator.DQLModelGenerator(config, features_names)
    
    stats = dql._deep_q_learning()
    q_network = dql.q_network
    
    with open("/clusterlivenfs/lcap/ids-online/Deep_Q_Learning_Auto_IDS/trained_models/dql.pkl", "wb") as file:
        pickle.dump(q_network, file)
        
    with open("/clusterlivenfs/lcap/ids-online/Deep_Q_Learning_Auto_IDS/trained_models/stats.pkl", "wb") as file:
        pickle.dump(stats, file)
    
    dql.test_model()
    
    
if __name__ == "__main__":
    main()