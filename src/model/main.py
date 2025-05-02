import model_generator
import json
import argparse
import torch
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
    features_names.extend(
+        f"feat_{i}" for i in range(config["config_model"]["feature_size"]))
        
    if args.mode == "Train":
        dql = model_generator.DQLModelGenerator(config, features_names)
        
        dql.deep_q_learning()
        
        torch.save(dql.q_network.state_dict(), f"{config["config_model"]["save_path"]}.pth")
                    
        dql.test_model()
    else:
        dql = model_generator.DQLModelGenerator(config, features_names)
        
        dql.q_network.load_state_dict(torch.load(f"{config["config_model"]["save_path"]}.pth"))
        
        dql.test_model()
        
        dql.save_metric(config)
        
if __name__ == "__main__":
    main()