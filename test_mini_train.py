"""
Mini training test - 500 steps with real environment interaction
Tests the complete training pipeline including logging
"""

import os
import torch
from config.ppo_config import PPOConfig
from src.environment import CarRacingEnv, NormalizeActions
from src.ppo_agent import PPOAgent
from src.utils import RolloutBuffer, MetricsLogger, set_seed

print("=" * 70)
print("MINI TRAINING TEST - 500 STEPS")
print("=" * 70)

# Configuration
config = PPOConfig()
config.n_steps = 256  # Steps before update
config.total_timesteps = 500  # Total training steps
config.device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seed
set_seed(config.seed)
print(f"\nUsing device: {config.device}")
print(f"Seed: {config.seed}")

# Create temp directories
import tempfile
temp_dir = tempfile.mkdtemp()
log_dir = os.path.join(temp_dir, "logs")
checkpoint_dir = os.path.join(temp_dir, "checkpoints")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Temp directory: {temp_dir}")

# Initialize environment
print("\n1. Initializing environment...")
env = NormalizeActions(CarRacingEnv())
obs_shape = env.observation_space.shape
action_dim = env.action_space.shape[0]
print(f"   Observation shape: {obs_shape}")
print(f"   Action dimension: {action_dim}")

# Initialize agent
print("\n2. Initializing agent...")
agent = PPOAgent(obs_shape, action_dim, config)
print(f"   Device: {agent.device}")
print(f"   Parameters: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")

# Initialize logger
print("\n3. Initializing logger...")
logger = MetricsLogger(log_dir)

# Training loop
print("\n4. Starting mini training loop...")
print(f"   Target: {config.total_timesteps} timesteps")

rollout_buffer = RolloutBuffer()
obs, _ = env.reset(seed=config.seed)
episode_reward = 0
episode_length = 0
n_episodes = 0
n_updates = 0

for timestep in range(1, config.total_timesteps + 1):
    # Get action
    action, log_prob, value = agent.get_action(obs)

    # Environment step
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Store transition
    rollout_buffer.add(obs, action, reward, value[0], log_prob, done)

    episode_reward += reward
    episode_length += 1
    obs = next_obs

    # Episode end
    if done or episode_length >= 1000:
        n_episodes += 1
        logger.log_episode(timestep, episode_reward, episode_length)
        print(f"   Episode {n_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")

        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

    # PPO update
    if len(rollout_buffer) >= config.n_steps:
        _, _, next_value = agent.get_action(obs)

        rollout_data = rollout_buffer.get()
        advantages, returns = agent.compute_gae(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones'],
            next_value[0]
        )

        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns

        losses = agent.update(rollout_data)
        logger.log_update(losses)

        rollout_buffer.clear()
        n_updates += 1

        print(f"   Update {n_updates}: Policy={losses['policy_loss']:.4f}, "
              f"Value={losses['value_loss']:.4f}, Entropy={losses['entropy']:.4f}")

# Save checkpoint
print("\n5. Saving checkpoint...")
checkpoint_path = os.path.join(checkpoint_dir, "mini_test.pt")
agent.save(checkpoint_path)
print(f"   Saved to: {checkpoint_path}")

# Verify checkpoint can be loaded
print("\n6. Verifying checkpoint load...")
agent2 = PPOAgent(obs_shape, action_dim, config)
agent2.load(checkpoint_path)
print("   Checkpoint loaded successfully")

# Save metrics
print("\n7. Saving metrics...")
metrics_path = os.path.join(log_dir, "metrics.npz")
logger.save_metrics(metrics_path)
print(f"   Metrics saved to: {metrics_path}")

# Cleanup
env.close()

# Print summary
print("\n" + "=" * 70)
print("MINI TRAINING TEST RESULTS")
print("=" * 70)
print(f"Total timesteps: {timestep}")
print(f"Episodes completed: {n_episodes}")
print(f"PPO updates: {n_updates}")
print(f"Average episode reward: {sum(logger.episode_rewards)/len(logger.episode_rewards):.2f}")
print(f"Average episode length: {sum(logger.episode_lengths)/len(logger.episode_lengths):.1f}")

# Cleanup temp directory
import shutil
shutil.rmtree(temp_dir)
print(f"\nTemp directory cleaned: {temp_dir}")

print("\n" + "=" * 70)
print(">>> MINI TRAINING TEST SUCCESSFUL <<<")
print("=" * 70)
print("\nComplete training pipeline verified:")
print("  - Environment interaction")
print("  - Agent updates")
print("  - Logging system")
print("  - Checkpoint system")
print("  - All components working together")
