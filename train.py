"""
Training script for PPO on Car Racing
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from config.ppo_config import PPOConfig
from src.environment import CarRacingEnv, NormalizeActions
from src.ppo_agent import PPOAgent
from src.utils import set_seed, RolloutBuffer, MetricsLogger, evaluate_agent


def train():
    """Main training loop"""
    # Configuration
    config = PPOConfig()

    # Set seed for reproducibility
    set_seed(config.seed)

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize environment
    env = NormalizeActions(CarRacingEnv(render_mode=config.render_mode))
    eval_env = NormalizeActions(CarRacingEnv())

    # Initialize agent
    obs_shape = env.observation_space.shape  # (frame_stack, 84, 84)
    action_dim = env.action_space.shape[0]   # 3 (steering, gas, brake)

    agent = PPOAgent(obs_shape, action_dim, config)

    # Logger
    logger = MetricsLogger("logs")

    print("=" * 50)
    print("PPO Training - Car Racing")
    print("=" * 50)
    print(f"Device: {agent.device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print("=" * 50)
    print()

    # Training loop
    rollout_buffer = RolloutBuffer()
    obs, _ = env.reset(seed=config.seed)
    episode_reward = 0
    episode_length = 0
    n_episodes = 0
    n_updates = 0
    best_eval_reward = -float('inf')

    for timestep in tqdm(range(1, config.total_timesteps + 1), desc="Training"):
        # Collect rollout
        action, log_prob, value = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        rollout_buffer.add(obs, action, reward, value[0], log_prob, done)

        episode_reward += reward
        episode_length += 1

        obs = next_obs

        # Episode end
        if done or episode_length >= config.max_episode_steps:
            n_episodes += 1
            logger.log_episode(timestep, episode_reward, episode_length)

            # Reset
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        # PPO update
        if len(rollout_buffer) >= config.n_steps:
            # Get next value for GAE
            _, _, next_value = agent.get_action(obs)

            # Compute advantages and returns
            rollout_data = rollout_buffer.get()
            advantages, returns = agent.compute_gae(
                rollout_data['rewards'],
                rollout_data['values'],
                rollout_data['dones'],
                next_value[0]
            )

            # Add to rollout data
            rollout_data['advantages'] = advantages
            rollout_data['returns'] = returns

            # PPO update
            losses = agent.update(rollout_data)
            logger.log_update(losses)

            # Clear buffer
            rollout_buffer.clear()
            n_updates += 1

            # Logging
            if n_updates % config.log_interval == 0 and logger.episode_rewards:
                recent_rewards = logger.episode_rewards[-100:]
                recent_lengths = logger.episode_lengths[-100:]
                mean_reward = np.mean(recent_rewards)
                mean_length = np.mean(recent_lengths)

                logger.print_progress(timestep, n_episodes, mean_reward, mean_length)

            # Evaluation
            if n_updates % config.eval_interval == 0:
                print("\nEvaluating...")
                eval_reward, eval_std, eval_length = evaluate_agent(
                    agent, eval_env, config.eval_episodes
                )
                print(f"Eval - Mean Reward: {eval_reward:.2f} +/- {eval_std:.2f}, "
                      f"Mean Length: {eval_length:.1f}\n")

                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save("checkpoints/best_model.pt")
                    print(f"New best model saved! Reward: {eval_reward:.2f}\n")

            # Save checkpoint
            if n_updates % config.save_interval == 0:
                agent.save(f"checkpoints/checkpoint_{n_updates}.pt")

    # Final save
    agent.save("checkpoints/final_model.pt")
    logger.save_metrics("logs/training_metrics.npz")
    logger.plot_metrics("logs/training_curves.png")

    # Final evaluation
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print("\nFinal Evaluation...")
    eval_reward, eval_std, eval_length = evaluate_agent(agent, eval_env, 10)
    print(f"Final - Mean Reward: {eval_reward:.2f} +/- {eval_std:.2f}, "
          f"Mean Length: {eval_length:.1f}")

    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
