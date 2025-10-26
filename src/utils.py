"""
Utility functions for training, logging, and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
from datetime import datetime


def set_seed(seed):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class RolloutBuffer:
    """
    Buffer to store rollout data for PPO update
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, reward, value, log_prob, done):
        """Add transition to buffer"""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        """Get all data from buffer"""
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': np.array(self.log_probs),
            'dones': self.dones
        }

    def clear(self):
        """Clear buffer"""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.obs)


class MetricsLogger:
    """
    Logger for training metrics
    """

    def __init__(self, log_dir):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.timesteps = []

        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write("=" * 50 + "\n\n")

    def log_episode(self, timestep, episode_reward, episode_length):
        """Log episode metrics"""
        self.timesteps.append(timestep)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

    def log_update(self, losses):
        """Log update metrics"""
        self.policy_losses.append(losses['policy_loss'])
        self.value_losses.append(losses['value_loss'])
        self.entropies.append(losses['entropy'])

    def print_progress(self, timestep, n_episodes, mean_reward, mean_length):
        """Print training progress"""
        msg = (
            f"Timestep: {timestep:,} | "
            f"Episodes: {n_episodes} | "
            f"Mean Reward: {mean_reward:.2f} | "
            f"Mean Length: {mean_length:.1f}"
        )
        print(msg)

        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")

    def plot_metrics(self, save_path=None):
        """
        Plot training metrics

        Args:
            save_path: Path to save plot (if None, just display)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)

        # Episode rewards
        axes[0, 0].plot(self.timesteps, self.episode_rewards, alpha=0.6)
        axes[0, 0].plot(self.timesteps, self._moving_average(self.episode_rewards, 10), linewidth=2, label='MA(10)')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(self.timesteps, self.episode_lengths, alpha=0.6)
        axes[0, 1].plot(self.timesteps, self._moving_average(self.episode_lengths, 10), linewidth=2, label='MA(10)')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Policy loss
        if self.policy_losses:
            axes[1, 0].plot(self.policy_losses)
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Policy Loss')
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].grid(True)

        # Value loss
        if self.value_losses:
            axes[1, 1].plot(self.value_losses)
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Value Loss')
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to: {save_path}")
        else:
            plt.show()

    def _moving_average(self, data, window):
        """Calculate moving average"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')

    def save_metrics(self, save_path):
        """Save metrics to file"""
        np.savez(
            save_path,
            timesteps=self.timesteps,
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            policy_losses=self.policy_losses,
            value_losses=self.value_losses,
            entropies=self.entropies
        )
        print(f"Metrics saved to: {save_path}")


def evaluate_agent(agent, env, n_episodes=5):
    """
    Evaluate agent performance

    Args:
        agent: PPO agent
        env: Environment
        n_episodes: Number of evaluation episodes
    Returns:
        mean_reward, std_reward, mean_length
    """
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    return mean_reward, std_reward, mean_length
