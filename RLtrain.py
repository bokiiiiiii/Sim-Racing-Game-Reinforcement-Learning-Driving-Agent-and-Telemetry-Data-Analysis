import os
import time
from ACCEnv import ACCEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env


def train_agent():
    """
    Main function to initialize, train, and save the PPO agent for ACC.
    Allows for continuing training from a previously saved model.
    """
    # --- Configuration ---
    LOG_DIR = "logs/"
    MODEL_DIR = "models/"

    # Training hyperparameters
    TOTAL_TIMESTEPS = 1000000  # Total number of training steps
    SAVE_FREQ = 50000  # Save a checkpoint every n steps
    MODEL_NAME_PREFIX = "ppo_acc"  # Prefix for saved models and logs

    # Path for continuing training. If this file exists, training will resume from it.
    # To continue training, copy your desired model to this path or update the path.
    LOAD_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_acc_to_continue.zip")

    # PPO agent hyperparameters
    PPO_PARAMS = {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": "auto",
        # "policy_kwargs": dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]) # Example
    }

    # Create log and model save directories if they don't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Initialize ACC Environment
    print("Initializing ACC Environment...")
    try:
        env = ACCEnv()
        # Check if the environment follows the Gym API (optional but recommended)
        # check_env(env)  # May cause errors due to ACC's specificity, can be commented out if problematic
        # Commented out check_env to reduce initial resets.
        print("ACC Environment initialized successfully. (check_env skipped)")
    except Exception as e:
        print(f"Failed to initialize ACC Environment: {e}")
        print(
            "Please ensure Assetto Corsa Competizione is running and Shared Memory is enabled."
        )
        return

    # 2. Initialize PPO Agent or load existing one
    print("Initializing PPO Agent...")
    try:
        if os.path.exists(LOAD_MODEL_PATH):
            print(f"Loading existing model from: {LOAD_MODEL_PATH}")
            # When loading a model to continue training, ensure 'env' and 'tensorboard_log' are passed.
            # You can also adjust hyperparameters after loading, e.g., model.learning_rate.
            model = PPO.load(
                LOAD_MODEL_PATH,
                env=env,
                tensorboard_log=LOG_DIR,
                # Custom PPO.load arguments can be added here if needed, e.g., device="cuda"
            )
            # Example: Adjust learning rate for fine-tuning
            # model.learning_rate = 0.00005
            print("Model loaded successfully. Continuing training.")
        else:
            print(f"Model not found at '{LOAD_MODEL_PATH}'. Creating a new model.")
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=0,  # Changed from 1 to 0 to reduce console output
                tensorboard_log=LOG_DIR,
                **PPO_PARAMS,  # Unpack PPO hyperparameters
            )
            print("New PPO Agent initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize or load PPO Agent: {e}")
        env.close()
        return

    # 3. Set up callback to save model periodically
    current_timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_save_path = os.path.join(
        MODEL_DIR, f"{MODEL_NAME_PREFIX}_{current_timestamp}"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=checkpoint_save_path,
        name_prefix=MODEL_NAME_PREFIX,
    )

    # 4. Start training
    print(f"Starting training, total timesteps: {TOTAL_TIMESTEPS}...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=1,  # Log training progress every episode
            tb_log_name=f"{MODEL_NAME_PREFIX}_{current_timestamp}",  # For TensorBoard
        )
        print("Training completed.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # 5. Save final model
        final_model_name = f"{MODEL_NAME_PREFIX}_final_{current_timestamp}.zip"
        final_model_path = os.path.join(MODEL_DIR, final_model_name)
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # 6. Close environment
        print("Closing ACC Environment...")
        env.close()
        print("ACC Environment closed.")


if __name__ == "__main__":
    print("=" * 30)
    print("ACC Reinforcement Learning Training Script")
    print("=" * 30)
    print("Important Notes:")
    print("1. Ensure Assetto Corsa Competizione (ACC) game is running.")
    print("2. Ensure Shared Memory is enabled in ACC.")
    print("3. Ensure vJoy driver is installed and configured.")
    print("4. Training can take a long time, please be patient.")
    print("5. Press Ctrl+C to terminate training early and save the current model.")
    print("-" * 30)

    # Wait a few seconds to allow user to switch to game window or prepare
    wait_time = 5
    print(
        f"Training will start in {wait_time} seconds, please prepare the ACC game environment..."
    )
    for i in range(wait_time, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Starting training execution...")

    train_agent()

    print("=" * 30)
    print("Training script execution finished.")
    print("=" * 30)
