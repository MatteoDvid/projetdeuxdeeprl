"""
Test all imports to ensure no circular dependencies or missing modules
"""

print("Testing all imports...")
print("=" * 60)

# Core dependencies
print("\n1. Testing core dependencies...")
import torch
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")

import numpy as np
print(f"   NumPy: {np.__version__}")

import gymnasium as gym
print(f"   Gymnasium: OK")

# Box2D and pygame
print("\n2. Testing Box2D and pygame...")
import Box2D
print(f"   Box2D: OK")

import pygame
print(f"   pygame: {pygame.version.ver}")

# Visualization
print("\n3. Testing visualization libraries...")
import matplotlib
print(f"   Matplotlib: {matplotlib.__version__}")

import seaborn
print(f"   Seaborn: OK")

import tensorboard
print(f"   TensorBoard: OK")

# Image processing
print("\n4. Testing image processing...")
from PIL import Image
print(f"   Pillow: OK")

import cv2
print(f"   OpenCV: {cv2.__version__}")

# Project modules
print("\n5. Testing project modules...")
from config.ppo_config import PPOConfig
print(f"   config.ppo_config: OK")

from src.ppo_agent import PPOAgent, ActorCritic
print(f"   src.ppo_agent: OK")

from src.environment import CarRacingEnv, NormalizeActions
print(f"   src.environment: OK")

from src.utils import RolloutBuffer, MetricsLogger, set_seed, evaluate_agent
print(f"   src.utils: OK")

# Test environment creation
print("\n6. Testing environment creation...")
env = gym.make("CarRacing-v3", continuous=True)
print(f"   CarRacing-v3 environment: OK")
env.close()

wrapped_env = NormalizeActions(CarRacingEnv())
print(f"   Wrapped environment: OK")
wrapped_env.close()

# Test agent creation
print("\n7. Testing agent creation...")
config = PPOConfig()
config.device = "cuda" if torch.cuda.is_available() else "cpu"
agent = PPOAgent((4, 84, 84), 3, config)
print(f"   PPO Agent on {agent.device}: OK")

print("\n" + "=" * 60)
print("ALL IMPORTS SUCCESSFUL!")
print("=" * 60)
