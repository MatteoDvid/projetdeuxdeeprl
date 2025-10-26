"""
PPO Hyperparameter Configuration for Car Racing
"""

class PPOConfig:
    # Environment
    env_name = "CarRacing-v3"
    render_mode = None  # Set to "human" for visualization during training

    # Training
    total_timesteps = 1_000_000  # Total training steps
    max_episode_steps = 1000     # Maximum steps per episode

    # PPO Hyperparameters
    learning_rate = 3e-4         # Learning rate for optimizer
    gamma = 0.99                 # Discount factor
    gae_lambda = 0.95            # GAE lambda for advantage estimation
    clip_epsilon = 0.2           # PPO clipping parameter
    value_coef = 0.5             # Value loss coefficient
    entropy_coef = 0.01          # Entropy bonus coefficient
    max_grad_norm = 0.5          # Gradient clipping

    # Training Loop
    n_steps = 2048               # Steps to collect before update
    batch_size = 64              # Minibatch size
    n_epochs = 10                # Number of epochs per update

    # Network Architecture
    hidden_dims = [256, 256]     # Hidden layer dimensions

    # Logging and Saving
    log_interval = 10            # Log every N updates
    save_interval = 50           # Save checkpoint every N updates
    eval_interval = 25           # Evaluate every N updates
    eval_episodes = 5            # Number of episodes for evaluation

    # Device
    device = "cuda"              # "cuda" or "cpu"

    # Seed
    seed = 42

    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
