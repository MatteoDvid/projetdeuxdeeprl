"""
Complete Setup Verification Script
Tests all components before training
"""

import sys
import traceback


def test_imports():
    """Test 1: Verify all imports."""
    print("=" * 70)
    print("TEST 1: IMPORTS")
    print("=" * 70)

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"ERROR PyTorch: {e}")
        return False

    try:
        import gymnasium
        print(f"Gymnasium imported")
    except Exception as e:
        print(f"ERROR Gymnasium: {e}")
        return False

    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except Exception as e:
        print(f"ERROR NumPy: {e}")
        return False

    try:
        from PIL import Image
        print(f"Pillow imported")
    except Exception as e:
        print(f"ERROR Pillow: {e}")
        return False

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"ERROR OpenCV: {e}")
        return False

    print("\nAll imports OK!\n")
    return True


def test_environment():
    """Test 2: Verify Car Racing environment."""
    print("=" * 70)
    print("TEST 2: CAR RACING ENVIRONMENT")
    print("=" * 70)

    try:
        import gymnasium as gym

        # Create environment
        env = gym.make("CarRacing-v3", continuous=True)
        print("Environment created")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space.shape}")

        # Test reset
        obs, info = env.reset()
        print("Reset OK")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")
        print(f"  Observation range: [{obs.min():.1f}, {obs.max():.1f}]")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step OK")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        env.close()
        print("\nEnvironment works!\n")
        return True

    except Exception as e:
        print(f"\nERROR environment: {e}")
        traceback.print_exc()
        return False


def test_environment_wrapper():
    """Test 3: Verify environment wrapper."""
    print("=" * 70)
    print("TEST 3: ENVIRONMENT WRAPPER")
    print("=" * 70)

    try:
        from src.environment import CarRacingEnv, NormalizeActions
        import numpy as np

        # Create wrapped environment
        env = NormalizeActions(CarRacingEnv())
        print("Wrapped environment created")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")

        # Test reset
        obs, info = env.reset(seed=42)
        print("Reset OK")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Expected: (4, 84, 84) [frame_stack, height, width]")
        assert obs.shape == (4, 84, 84), f"Wrong shape: {obs.shape}"
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        assert 0 <= obs.min() and obs.max() <= 1, "Observations not normalized!"

        # Test step
        action = np.array([0.0, 0.5, 0.0])  # [steering, gas, brake]
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step OK")
        print(f"  Reward: {reward:.3f}")
        print(f"  Observation shape: {obs.shape}")
        assert obs.shape == (4, 84, 84), f"Wrong shape after step: {obs.shape}"

        env.close()
        print("\nEnvironment wrapper works!\n")
        return True

    except Exception as e:
        print(f"\nERROR environment wrapper: {e}")
        traceback.print_exc()
        return False


def test_ppo_agent():
    """Test 4: Verify PPO agent."""
    print("=" * 70)
    print("TEST 4: PPO AGENT")
    print("=" * 70)

    try:
        import torch
        from config.ppo_config import PPOConfig
        from src.ppo_agent import PPOAgent

        # Create config
        config = PPOConfig()
        config.device = "cpu"  # Force CPU for testing
        print(f"Config created")

        # Create agent
        obs_shape = (4, 84, 84)
        action_dim = 3
        agent = PPOAgent(obs_shape, action_dim, config)
        print(f"Agent created on {agent.device}")
        print(f"  Network parameters: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")

        # Test action selection
        import numpy as np
        dummy_obs = np.random.rand(4, 84, 84).astype(np.float32)
        action, log_prob, value = agent.get_action(dummy_obs)
        print("Action selection OK")
        print(f"  Action shape: {action.shape}")
        print(f"  Action: {action}")
        print(f"  Log prob: {log_prob}")
        print(f"  Value: {value}")

        # Test deterministic action
        action_det, _, _ = agent.get_action(dummy_obs, deterministic=True)
        print("Deterministic action OK")
        print(f"  Action: {action_det}")

        print("\nPPO agent works!\n")
        return True

    except Exception as e:
        print(f"\nERROR PPO agent: {e}")
        traceback.print_exc()
        return False


def test_training_components():
    """Test 5: Verify training utilities."""
    print("=" * 70)
    print("TEST 5: TRAINING UTILITIES")
    print("=" * 70)

    try:
        from src.utils import RolloutBuffer, MetricsLogger, set_seed
        import tempfile
        import os

        # Test seed setting
        set_seed(42)
        print("Seed setting OK")

        # Test rollout buffer
        buffer = RolloutBuffer()
        for i in range(10):
            buffer.add(
                obs=[[i]],
                action=[0.1 * i],
                reward=1.0,
                value=0.5,
                log_prob=0.1,
                done=False
            )
        print(f"Rollout buffer OK (length: {len(buffer)})")

        data = buffer.get()
        print(f"  Data keys: {list(data.keys())}")
        buffer.clear()
        print(f"  Cleared (length: {len(buffer)})")

        # Test metrics logger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            logger.log_episode(100, 50.0, 100)
            logger.log_update({'policy_loss': 0.1, 'value_loss': 0.2, 'entropy': 0.01})
            print("Metrics logger OK")
            print(f"  Episode rewards: {logger.episode_rewards}")
            print(f"  Policy losses: {logger.policy_losses}")

        print("\nTraining utilities work!\n")
        return True

    except Exception as e:
        print(f"\nERROR training utilities: {e}")
        traceback.print_exc()
        return False


def test_full_integration():
    """Test 6: Full integration test."""
    print("=" * 70)
    print("TEST 6: FULL INTEGRATION")
    print("=" * 70)

    try:
        import torch
        import numpy as np
        from config.ppo_config import PPOConfig
        from src.environment import CarRacingEnv, NormalizeActions
        from src.ppo_agent import PPOAgent
        from src.utils import RolloutBuffer

        # Setup
        config = PPOConfig()
        config.device = "cpu"  # Force CPU for testing

        env = NormalizeActions(CarRacingEnv())
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]

        agent = PPOAgent(obs_shape, action_dim, config)
        buffer = RolloutBuffer()

        print("Components initialized")

        # Run a few steps
        obs, _ = env.reset(seed=42)
        total_reward = 0

        for step in range(10):
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, value[0], log_prob, done)
            total_reward += reward

            obs = next_obs

            if done:
                obs, _ = env.reset()

        print(f"10 steps completed")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Buffer length: {len(buffer)}")

        # Test GAE computation
        rollout_data = buffer.get()
        _, _, next_value = agent.get_action(obs)
        advantages, returns = agent.compute_gae(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones'],
            next_value[0]
        )
        print(f"GAE computation OK")
        print(f"  Advantages length: {len(advantages)}")
        print(f"  Returns length: {len(returns)}")

        env.close()
        print("\nFull integration works!\n")
        return True

    except Exception as e:
        print(f"\nERROR integration: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("COMPLETE SETUP VERIFICATION")
    print("=" * 70 + "\n")

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Environment Wrapper", test_environment_wrapper()))
    results.append(("PPO Agent", test_ppo_agent()))
    results.append(("Training Utilities", test_training_components()))
    results.append(("Full Integration", test_full_integration()))

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False

    print("=" * 70 + "\n")

    if all_passed:
        print("ALL TESTS PASSED")
        print("\nYou can now train the model:")
        print("  python train.py")
        return 0
    else:
        print("SOME TESTS FAILED")
        print("\nDo NOT start training until all tests pass!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
