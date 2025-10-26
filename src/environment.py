"""
Environment wrapper for Car Racing
Handles preprocessing, frame stacking, and reward shaping
"""

import gymnasium as gym
import numpy as np
import torch
from collections import deque


class CarRacingEnv:
    """
    Wrapper for CarRacing-v3 environment with preprocessing
    """

    def __init__(self, render_mode=None, frame_skip=2, frame_stack=4):
        """
        Args:
            render_mode: "human" for visualization, None for training
            frame_skip: Number of frames to skip (action repeat)
            frame_stack: Number of frames to stack for temporal information
        """
        self.env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

        # Action space: [steering, gas, brake]
        # steering: [-1, 1], gas: [0, 1], brake: [0, 1]
        self.action_space = self.env.action_space

        # Observation space: stacked grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(frame_stack, 84, 84),
            dtype=np.float32
        )

    def preprocess_frame(self, frame):
        """
        Preprocess frame: grayscale, crop, resize, normalize

        Args:
            frame: RGB frame (96, 96, 3)
        Returns:
            Preprocessed grayscale frame (84, 84)
        """
        # Convert to grayscale
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

        # Crop bottom (remove car dashboard)
        gray = gray[:84, :]

        # Ensure uint8 for PIL
        gray = gray.astype(np.uint8)

        # Resize to 84x84
        from PIL import Image
        gray = np.array(Image.fromarray(gray).resize((84, 84)))

        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0

        return gray

    def reset(self, seed=None):
        """Reset environment and return initial observation"""
        obs, info = self.env.reset(seed=seed)

        # Preprocess and stack frames
        processed = self.preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)

        return np.array(self.frames), info

    def step(self, action):
        """
        Take action in environment with frame skipping

        Args:
            action: [steering, gas, brake]
        Returns:
            observation, reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Frame skip: repeat action and accumulate rewards
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Preprocess and add to frame stack
        processed = self.preprocess_frame(obs)
        self.frames.append(processed)

        # Reward shaping: penalize going off-track heavily
        if total_reward < 0:
            total_reward *= 2.0  # Amplify negative rewards

        return np.array(self.frames), total_reward, terminated, truncated, info

    def close(self):
        """Close environment"""
        self.env.close()

    def render(self):
        """Render environment"""
        return self.env.render()


class NormalizeActions:
    """
    Wrapper to normalize actions to [-1, 1] range
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)

        # Convert to environment action format
        # steering: [-1, 1], gas: [0, 1], brake: [0, 1]
        env_action = np.array([
            action[0],  # steering
            max(0, action[1]),  # gas
            max(0, action[2])   # brake
        ])

        return self.env.step(env_action)

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
