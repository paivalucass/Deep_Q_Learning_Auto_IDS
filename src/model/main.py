import model_generator
import json
import argparse
import torch
import matplotlib.pyplot as plt
import wandb

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
    
    # Initialize wandb
    wandb.init(
        project=config["config_model"].get("wandb_project", "DQL-IDS"),
        name=config["config_model"].get("wandb_run_name", f"{args.mode}_DQN"),
        config=config["config_model"]
    )

    features_names = [f"feat_{i}" for i in range(config["config_model"]["feature_size"])]
        
    dql = model_generator.DQLModelGenerator(config, features_names)
    
    if args.mode == "Train":
        stats = dql.deep_q_learning()

        # Log training stats to wandb
        for step, (loss, ret) in enumerate(zip(stats["MSE Loss"], stats["Returns"])):
            wandb.log({"MSE Loss": loss, "Return": ret, "Step": step})

        # Save model to wandb
        model_path = f"{config['config_model']['save_path']}.pth"
        torch.save(dql.q_network.state_dict(), model_path)
        wandb.save(model_path)

        dql.test_model()

    else:
        model_path = f"{config['config_model']['save_path']}.pth"
        dql.q_network.load_state_dict(torch.load(model_path))
        dql.test_model()
        dql.save_metric(config)

    wandb.finish()

if __name__ == "__main__":
    main()
