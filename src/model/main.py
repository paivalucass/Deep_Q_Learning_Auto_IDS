import online_deep_q_network
import json
import argparse
import torch
import matplotlib.pyplot as plt
import wandb
import datetime

def main():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--config', required=True, help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--mode', required=True, help='Train or Test mode')
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
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize wandb
    wandb.init(
        project=config["config_model"].get("wandb_project", "DQL-IDS"),
        name=config["config_model"].get("wandb_run_name", f"{args.mode}_DQN_{timestamp}"),
        config=config["config_model"]
    )

    features_names = [f"feat_{i}" for i in range(config["config_model"]["feature_size"])]
        
    dql = online_deep_q_network.DQNModelGenerator(config, features_names)
    
    if args.mode == "Train":
        stats = dql.deep_q_learning()

        # Save model to wandb
        model_path = f"{config['config_model']['save_path']}.pth"
        torch.save(dql.q_network.state_dict(), model_path)
        wandb.save(model_path)

        dql.test_model()

    else:
        model_path = f"{config['config_model']['save_path']}.pth"
        print(f"Loading model from: {model_path}")
        dql.q_network.load_state_dict(torch.load(model_path))
        dql.test_model()

    wandb.finish()

if __name__ == "__main__":
    main()
