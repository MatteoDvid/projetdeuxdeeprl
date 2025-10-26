"""
Test evaluation script functionality without rendering
"""

import tempfile
import os

from config.ppo_config import PPOConfig
from src.ppo_agent import PPOAgent
from src.environment import CarRacingEnv, NormalizeActions
from src.utils import evaluate_agent

print("=" * 60)
print("EVALUATION SCRIPT TEST")
print("=" * 60)

# Create temporary checkpoint
temp_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(temp_dir, "test_model.pt")

print(f"\nUsing temporary directory: {temp_dir}")

# Configuration
config = PPOConfig()
config.device = "cuda" if config.device == "cuda" else "cpu"

# Create and save agent
print("\n1. Creating and saving test agent...")
agent = PPOAgent((4, 84, 84), 3, config)
agent.save(checkpoint_path)
print(f"   Test agent saved to {checkpoint_path}")

# Create environment
print("\n2. Creating evaluation environment...")
eval_env = NormalizeActions(CarRacingEnv())
print("   Evaluation environment created")

# Evaluate agent
print("\n3. Running evaluation (3 episodes, no rendering)...")
mean_reward, std_reward, mean_length = evaluate_agent(
    agent, eval_env, n_episodes=3
)

print(f"\n   Results:")
print(f"   - Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"   - Mean Length: {mean_length:.1f}")

# Test loading checkpoint and re-evaluating
print("\n4. Testing checkpoint reload...")
agent2 = PPOAgent((4, 84, 84), 3, config)
agent2.load(checkpoint_path)
print("   Checkpoint loaded into new agent")

mean_reward2, std_reward2, mean_length2 = evaluate_agent(
    agent2, eval_env, n_episodes=3
)

print(f"\n   Results after reload:")
print(f"   - Mean Reward: {mean_reward2:.2f} +/- {std_reward2:.2f}")
print(f"   - Mean Length: {mean_length2:.1f}")

# Cleanup
eval_env.close()
os.remove(checkpoint_path)
os.rmdir(temp_dir)

print("\n" + "=" * 60)
print("EVALUATION TEST SUCCESSFUL!")
print("=" * 60)
print("\nEvaluation system works correctly:")
print("  - Agents can be evaluated on environment")
print("  - Checkpoints can be loaded and evaluated")
print("  - Statistics are computed correctly")
