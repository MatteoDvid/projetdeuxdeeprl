"""
Quick training test to verify the training loop works
Runs for just a few steps to ensure everything is connected properly
"""

import torch
from config.ppo_config import PPOConfig
from src.environment import CarRacingEnv, NormalizeActions
from src.ppo_agent import PPOAgent
from src.utils import RolloutBuffer, set_seed

print("=" * 60)
print("QUICK TRAINING TEST")
print("=" * 60)

# Configuration with reduced steps for testing
config = PPOConfig()
config.n_steps = 128  # Reduced from 2048
config.device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seed
set_seed(config.seed)

# Initialize environment
print("\nInitializing environment...")
env = NormalizeActions(CarRacingEnv())
obs_shape = env.observation_space.shape
action_dim = env.action_space.shape[0]

print(f"  Observation shape: {obs_shape}")
print(f"  Action dimension: {action_dim}")

# Initialize agent
print("\nInitializing PPO agent...")
agent = PPOAgent(obs_shape, action_dim, config)
print(f"  Device: {agent.device}")
print(f"  Network parameters: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")

# Rollout buffer
rollout_buffer = RolloutBuffer()

# Run a short training loop
print("\nRunning short training loop...")
print(f"  Collecting {config.n_steps} steps...")

obs, _ = env.reset(seed=config.seed)
episode_reward = 0
episode_length = 0
n_episodes = 0

for step in range(1, config.n_steps + 1):
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
        print(f"  Episode {n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

print(f"\nCollected {len(rollout_buffer)} transitions")

# Test PPO update
print("\nTesting PPO update...")
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

print(f"  Policy loss: {losses['policy_loss']:.4f}")
print(f"  Value loss: {losses['value_loss']:.4f}")
print(f"  Entropy: {losses['entropy']:.4f}")

# Cleanup
env.close()

print("\n" + "=" * 60)
print("TRAINING TEST SUCCESSFUL!")
print("=" * 60)
print("\nThe training loop works correctly.")
print("You can now run full training with:")
print("  python train.py")
print("=" * 60)
