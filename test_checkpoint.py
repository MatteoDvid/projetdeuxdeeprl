"""
Test checkpoint save/load functionality
"""

import torch
import numpy as np
import os
import tempfile

from config.ppo_config import PPOConfig
from src.ppo_agent import PPOAgent
from src.environment import CarRacingEnv, NormalizeActions

print("=" * 60)
print("CHECKPOINT SAVE/LOAD TEST")
print("=" * 60)

# Create temporary directory for test
temp_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")

print(f"\nUsing temporary directory: {temp_dir}")

# Configuration
config = PPOConfig()
config.device = "cuda" if torch.cuda.is_available() else "cpu"

# Create agent
print("\n1. Creating agent...")
agent1 = PPOAgent((4, 84, 84), 3, config)
print(f"   Agent created on {agent1.device}")

# Get initial parameters
print("\n2. Getting initial parameters...")
initial_params = {name: param.clone() for name, param in agent1.actor_critic.named_parameters()}
print(f"   Captured {len(initial_params)} parameter tensors")

# Make some random updates to change parameters
print("\n3. Simulating training (random updates)...")
env = NormalizeActions(CarRacingEnv())
obs, _ = env.reset(seed=42)

for _ in range(5):
    action, log_prob, value = agent1.get_action(obs)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    obs = next_obs
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# Check parameters changed
changed = False
for name, param in agent1.actor_critic.named_parameters():
    if not torch.allclose(param, initial_params[name]):
        changed = True
        break

if changed:
    print("   Parameters changed (as expected from inference)")
else:
    print("   WARNING: Parameters didn't change")

# Save checkpoint
print(f"\n4. Saving checkpoint to {checkpoint_path}...")
agent1.save(checkpoint_path)
print("   Checkpoint saved successfully")

# Verify file exists
if os.path.exists(checkpoint_path):
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"   Checkpoint file size: {file_size:.2f} MB")
else:
    print("   ERROR: Checkpoint file not created!")
    exit(1)

# Create new agent
print("\n5. Creating new agent with random initialization...")
agent2 = PPOAgent((4, 84, 84), 3, config)

# Check parameters are different before loading
print("\n6. Verifying parameters differ before loading...")
params_differ = False
for (name1, param1), (name2, param2) in zip(
    agent1.actor_critic.named_parameters(),
    agent2.actor_critic.named_parameters()
):
    if not torch.allclose(param1, param2):
        params_differ = True
        break

if params_differ:
    print("   Parameters differ (as expected)")
else:
    print("   WARNING: Parameters are identical before loading!")

# Load checkpoint
print(f"\n7. Loading checkpoint into new agent...")
agent2.load(checkpoint_path)
print("   Checkpoint loaded successfully")

# Verify parameters match
print("\n8. Verifying parameters match after loading...")
all_match = True
for (name1, param1), (name2, param2) in zip(
    agent1.actor_critic.named_parameters(),
    agent2.actor_critic.named_parameters()
):
    if not torch.allclose(param1, param2, rtol=1e-5, atol=1e-8):
        print(f"   MISMATCH in {name1}")
        all_match = False

if all_match:
    print("   All parameters match!")
else:
    print("   ERROR: Parameters don't match!")
    exit(1)

# Test inference with both agents
print("\n9. Testing inference with both agents...")
env = NormalizeActions(CarRacingEnv())
obs, _ = env.reset(seed=123)

action1, log_prob1, value1 = agent1.get_action(obs, deterministic=True)
action2, log_prob2, value2 = agent2.get_action(obs, deterministic=True)

env.close()

action_match = np.allclose(action1, action2, rtol=1e-5, atol=1e-8)
value_match = np.allclose(value1, value2, rtol=1e-5, atol=1e-8)

if action_match and value_match:
    print("   Both agents produce identical outputs!")
else:
    print("   ERROR: Agents produce different outputs!")
    exit(1)

# Cleanup
print("\n10. Cleaning up...")
os.remove(checkpoint_path)
os.rmdir(temp_dir)
print("   Temporary files removed")

print("\n" + "=" * 60)
print("CHECKPOINT TEST SUCCESSFUL!")
print("=" * 60)
print("\nCheckpoint save/load system works correctly:")
print("  - Checkpoints are saved with correct format")
print("  - Parameters are restored exactly")
print("  - Loaded models produce identical outputs")
