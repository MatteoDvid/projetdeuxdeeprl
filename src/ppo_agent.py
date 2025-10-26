"""
PPO Agent Implementation for Car Racing
Includes Actor-Critic network, PPO algorithm, and training logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    - Actor: Policy network that outputs action distribution
    - Critic: Value network that estimates state value
    """

    def __init__(self, obs_shape, action_dim, hidden_dims=[256, 256]):
        """
        Args:
            obs_shape: Shape of observation (frame_stack, height, width)
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
        """
        super(ActorCritic, self).__init__()

        # Shared CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(obs_shape)

        # Actor (policy) network
        actor_layers = []
        prev_dim = conv_out_size
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.actor_mean = nn.Sequential(
            *actor_layers,
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Action standard deviation (learnable)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value) network
        critic_layers = []
        prev_dim = conv_out_size
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.critic = nn.Sequential(
            *critic_layers,
            nn.Linear(prev_dim, 1)
        )

    def _get_conv_out_size(self, shape):
        """Calculate output size of conv layers"""
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return x.shape[1]

    def forward(self, obs):
        """
        Forward pass through network

        Args:
            obs: Observation tensor
        Returns:
            action_mean, action_std, value
        """
        features = self.conv(obs)

        # Actor: action distribution
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)

        # Critic: state value
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy

        Args:
            obs: Observation tensor
            deterministic: If True, return mean action
        Returns:
            action, log_prob, value
        """
        action_mean, action_std, value = self.forward(obs)

        if deterministic:
            action = action_mean
            log_prob = None
        else:
            # Sample from Gaussian distribution
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for PPO update

        Args:
            obs: Observation tensor
            actions: Action tensor
        Returns:
            log_probs, values, entropy
        """
        action_mean, action_std, values = self.forward(obs)

        # Action distribution
        dist = Normal(action_mean, action_std)

        # Log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Entropy for exploration bonus
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values, entropy


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    """

    def __init__(self, obs_shape, action_dim, config):
        """
        Args:
            obs_shape: Shape of observation
            action_dim: Dimension of action space
            config: PPOConfig object
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Actor-Critic network
        self.actor_critic = ActorCritic(
            obs_shape, action_dim, config.hidden_dims
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.learning_rate
        )

        # Hyperparameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm

    def get_action(self, obs, deterministic=False):
        """Get action from policy"""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(obs, deterministic)

        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().numpy() if log_prob is not None else None
        value = value.cpu().numpy()[0]

        return action, log_prob, value

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            next_value: Value of next state
        Returns:
            advantages, returns
        """
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return advantages, returns

    def update(self, rollout_buffer):
        """
        PPO update step

        Args:
            rollout_buffer: Dictionary with rollout data
        Returns:
            losses: Dictionary of loss values
        """
        # Extract rollout data
        obs = torch.FloatTensor(rollout_buffer['obs']).to(self.device)
        actions = torch.FloatTensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout_buffer['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer['advantages']).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # Multiple epochs of updates
        for _ in range(self.config.n_epochs):
            # Mini-batch updates
            indices = np.arange(len(obs))
            np.random.shuffle(indices)

            for start in range(0, len(obs), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                # Batch data
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate actions with current policy
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
