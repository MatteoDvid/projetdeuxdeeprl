# Setup Instructions for Car Racing PPO Project

Complete step-by-step guide to set up the project on Windows.

## Prerequisites

- Python 3.11
- CUDA-capable GPU (optional, but recommended)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd car-racing-rl
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 4. Install PyTorch with CUDA Support

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision torchaudio
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important Note for Windows Users:**
- The `Box2D` package (capital B and D) provides precompiled wheels for Windows
- If you encounter issues, make sure you're installing `Box2D` not `box2d-py`

### 6. Verify Installation

Run the complete setup verification:
```bash
python test_setup.py
```

All tests should pass. You should see:
```
ALL TESTS PASSED

You can now train the model:
  python train.py
```

### 7. Quick Training Test (Optional)

Verify the training loop works:
```bash
python test_train.py
```

This runs a short training loop (128 steps) to ensure everything is connected properly.

## Common Issues and Solutions

### Issue: Box2D Installation Fails

**Solution:** Make sure you're installing `Box2D` (capital letters) not `box2d-py`:
```bash
pip install Box2D
```

### Issue: pygame Not Found

**Solution:** Install pygame explicitly:
```bash
pip install pygame
```

### Issue: CUDA Not Available

**Solution:** Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with the correct CUDA version for your system.

### Issue: Import Errors

**Solution:** Make sure you're in the project root directory and the virtual environment is activated.

## Training the Model

Once setup is complete, start training:

```bash
python train.py
```

Training parameters can be modified in `config/ppo_config.py`.

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Add `--no-render` to evaluate without visualization.

## Project Structure

```
car-racing-rl/
├── config/
│   └── ppo_config.py          # Hyperparameters
├── src/
│   ├── ppo_agent.py           # PPO algorithm
│   ├── environment.py         # Environment wrapper
│   └── utils.py               # Utilities
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── test_setup.py              # Setup verification
├── test_train.py              # Training test
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Next Steps

1. Review `config/ppo_config.py` to adjust hyperparameters
2. Run `python train.py` to start training
3. Monitor training progress in the `logs/` directory
4. Check saved models in `checkpoints/`
5. Analyze results using the Jupyter notebook in `notebooks/`

## Support

If you encounter any issues not covered here, please:
1. Check that all dependencies are correctly installed
2. Verify Python version (3.11)
3. Ensure CUDA drivers are up to date (if using GPU)
4. Review error messages carefully
