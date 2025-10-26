# Car Racing with PPO - Reinforcement Learning Project

Implementation of Proximal Policy Optimization (PPO) for the Car Racing environment.

## Overview

This project trains a reinforcement learning agent to drive a car around a racing track using PPO. The agent learns to control steering, acceleration, and braking from pixel observations.

## Environment

- **Name**: CarRacing-v3 (Gymnasium)
- **Action Space**: Continuous control [steering, gas, brake]
  - Steering: [-1, 1] (left to right)
  - Gas: [0, 1] (no acceleration to full acceleration)
  - Brake: [0, 1] (no braking to full braking)
- **Observation Space**: 96x96x3 RGB image
- **Reward Structure**:
  - +1000/N for every track tile visited (N = total tiles)
  - Negative reward for going off-track
  - Episode ends after 1000 steps or if car is stuck

## Algorithm: PPO

PPO is a policy gradient method that:
- Uses a clipped surrogate objective for stable updates
- Employs Generalized Advantage Estimation (GAE)
- Works well with continuous action spaces
- Balances exploration and exploitation

### Architecture

**Shared CNN Feature Extractor:**
- Conv2D(4→32, kernel=8, stride=4) + ReLU
- Conv2D(32→64, kernel=4, stride=2) + ReLU
- Conv2D(64→64, kernel=3, stride=1) + ReLU
- Flatten

**Actor Head (Policy):**
- FC(features→256) + ReLU
- FC(256→256) + ReLU
- FC(256→3) + Tanh
- Outputs: action means + learnable log standard deviations

**Critic Head (Value):**
- FC(features→256) + ReLU
- FC(256→256) + ReLU
- FC(256→1)
- Outputs: state value estimate

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- PyTorch 2.0+
- Gymnasium with Box2D
- NumPy, Matplotlib
- OpenCV, Pillow

## Project Structure

```
car-racing-rl/
├── config/
│   └── ppo_config.py          # Hyperparameters
├── src/
│   ├── ppo_agent.py           # PPO algorithm implementation
│   ├── environment.py         # Environment wrapper and preprocessing
│   └── utils.py               # Training utilities and logging
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── test_setup.py              # Setup verification tests
├── notebooks/
│   └── car_racing_ppo.ipynb   # Documented experiments
├── logs/                      # Training logs and metrics
└── checkpoints/               # Saved models
```

## Usage

### Verify Setup

Before training, verify that everything works:

```bash
python test_setup.py
```

### Training

Train the PPO agent:

```bash
python train.py
```

Training parameters (in `config/ppo_config.py`):
- Total timesteps: 1,000,000
- Learning rate: 3e-4
- Batch size: 64
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Clip epsilon: 0.2

### Evaluation

Evaluate a trained model:

```bash
# Evaluate with rendering
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate without rendering
python evaluate.py --checkpoint checkpoints/best_model.pt --no-render

# More evaluation episodes
python evaluate.py --checkpoint checkpoints/best_model.pt --episodes 10
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 3e-4 | Adam optimizer learning rate |
| Gamma | 0.99 | Discount factor for future rewards |
| GAE Lambda | 0.95 | Trade-off between bias and variance in advantage estimation |
| Clip Epsilon | 0.2 | PPO clipping parameter |
| Steps per Update | 2048 | Rollout steps before each PPO update |
| Batch Size | 64 | Mini-batch size for PPO updates |
| Epochs per Update | 10 | Number of optimization epochs per update |
| Value Coefficient | 0.5 | Weight for value loss in total loss |
| Entropy Coefficient | 0.01 | Weight for entropy bonus (exploration) |

## Preprocessing

The environment applies several preprocessing steps:
1. **Grayscale conversion**: RGB → Grayscale (reduces dimensionality)
2. **Cropping**: Remove bottom 12 pixels (car dashboard)
3. **Resizing**: 96x96 → 84x84
4. **Normalization**: Scale pixels to [0, 1]
5. **Frame stacking**: Stack 4 consecutive frames (temporal context)
6. **Frame skipping**: Repeat each action for 2 frames (action persistence)

## Results

Results will be documented after training completion:
- Training curves (rewards, losses)
- Final performance metrics
- Learned behaviors analysis

## References

- Schulman et al. (2017). "Proximal Policy Optimization Algorithms". arXiv:1707.06347
- Schulman et al. (2016). "High-Dimensional Continuous Control Using Generalized Advantage Estimation". arXiv:1506.02438
- Gymnasium Documentation: https://gymnasium.farama.org/
