"""
Evaluation script for trained PPO agent
"""

import argparse
import numpy as np
import time

from config.ppo_config import PPOConfig
from src.environment import CarRacingEnv, NormalizeActions
from src.ppo_agent import PPOAgent


def evaluate(checkpoint_path, n_episodes=5, render=True):
    """
    Evaluate trained agent

    Args:
        checkpoint_path: Path to model checkpoint
        n_episodes: Number of episodes to evaluate
        render: Whether to render environment
    """
    # Configuration
    config = PPOConfig()

    # Initialize environment
    render_mode = "human" if render else None
    env = NormalizeActions(CarRacingEnv(render_mode=render_mode))

    # Initialize agent
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_shape, action_dim, config)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)
    print("Checkpoint loaded successfully!")
    print()

    # Evaluate
    episode_rewards = []
    episode_lengths = []

    print("=" * 50)
    print(f"Evaluating for {n_episodes} episodes...")
    print("=" * 50)
    print()

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # Get action (deterministic)
            action, _, _ = agent.get_action(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                time.sleep(0.02)  # Slow down for visualization

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.1f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO agent on Car Racing")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")

    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, not args.no_render)
