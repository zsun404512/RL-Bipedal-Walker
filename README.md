# AISF Application - BipedalWalker RL Agent

This repository contains the implementation of a Reinforcement Learning agent to solve the `BipedalWalker-v3` environment as part of the AISF Application.

## Project Structure

- `bipedal_walker_rl.ipynb`: Main notebook with implementation, training loop, evaluation, and writeup.
- `report.md`: Summary/report notes for experiments.
- `OBSTACLES_README.md`: Notes related to obstacle settings and observations.
- `notes/`: Misc notes, e.g. `Note.md`.
- `logs/`: Training logs, checkpoints, and run outputs.
- `videos/`: Evaluation and demo videos.

## Requirements & Installation

Box2D requires `swig`. Install steps differ by platform:

### Colab (CUDA GPU)

Run these cells at the top of the notebook:

```bash
# System deps
apt-get update && apt-get install -y swig

# PyTorch with CUDA (adjust cu version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# RL and utilities
pip install "gymnasium[box2d]" numpy matplotlib ipython moviepy tqdm imageio[ffmpeg]
```

### macOS (Apple Silicon, MPS)

```bash
# System dep
brew install swig

# Optional: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# PyTorch with MPS support (from PyPI)
pip install torch torchvision torchaudio

# RL and utilities
pip install "gymnasium[box2d]" numpy matplotlib ipython moviepy tqdm imageio[ffmpeg]
```

If `gymnasium[box2d]` fails, ensure `swig` is installed, then try:

```bash
pip install box2d-py
```

## Hardware Acceleration

The code automatically detects and uses the available hardware acceleration:
- **macOS (Apple Silicon)**: Uses Metal Performance Shaders (`mps`) for GPU acceleration.
- **NVIDIA GPUs**: Uses CUDA (`cuda`) if available.
- **CPU**: Fallback if no GPU is detected.

## Environment Initialization

You can run multiple environments in parallel to speed up training. Recommended configurations:

- **Colab (CUDA GPU)**: Use 32 parallel environments.
- **macOS (MPS)**: Use 4 parallel environments for faster throughput.
- **Rendering enabled**: Use only 1 environment to avoid UI contention and slowdowns.

Example setup using Gymnasium's vector API:

```python
import torch
import gymnasium as gym

# Select device automatically
device = (
   "cuda" if torch.cuda.is_available() else
   ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)

def make_envs(n_envs: int, render: bool = False):
   render_mode = "human" if render else None
   env = gym.vector.make("BipedalWalker-v3", num_envs=n_envs, render_mode=render_mode)
   return env

# Recommended configurations
# Colab with CUDA GPU
env = make_envs(n_envs=32, render=False)

# macOS with MPS (Apple Silicon)
# env = make_envs(n_envs=4, render=False)

# Rendering-enabled run (use a single environment)
# env = make_envs(n_envs=1, render=True)
```

If your training loop or agent class expects a single-environment interface, consider using wrappers to adapt vectorized environments or switch to `num_envs=1`. For recording videos, prefer running evaluation with `num_envs=1` and `render_mode="human"` or use a video wrapper.

## Quick Start

### Notebook (Recommended)

1. Open `bipedal_walker_rl.ipynb`.
2. Install dependencies using the platform-specific steps above.
3. Set parallel environments per platform:
   - Colab (CUDA): 32
   - macOS (MPS): 4
   - Rendering: 1
4. Run cells to train, log metrics, and optionally record videos.

### Training & Evaluation Flow

- **Train**: Use vectorized envs (32 on Colab, 4 on macOS). Disable rendering during training.
- **Checkpointing**: Checkpoints (`*.pth`) are saved under `logs/<run_id>/`.
- **Evaluate & Render**: Switch to `num_envs=1` with `render_mode="human"` to visualize or record. Save videos under `videos/` or within a run-specific folder.

Example evaluation snippet:

```python
# Single-env evaluation with rendering
eval_env = gym.make("BipedalWalker-v3", render_mode="human")
# load checkpoint, run policy for N steps/episodes
```

## Troubleshooting

- **Box2D build errors**: Ensure `swig` is installed (`apt-get install -y swig` on Colab, `brew install swig` on macOS).
- **PyTorch device**: Confirm `device` resolves to `cuda` on Colab or `mps` on macOS; otherwise training will run on CPU.
- **Slow or choppy rendering**: Use `num_envs=1` and avoid rendering during training; render only for evaluation/demo.

## Usage

### Using Jupyter Notebook (Recommended)

1. Open `bipedal_walker_rl.ipynb`.
2. Run the cells in order to:
   - Initialize the Soft Actor-Critic (SAC) agent.
   - Train the agent in the `BipedalWalker-v3` environment.
   - Visualize the training progress.
   - Record and view a video of the trained agent.
   - Fill out the writeup section.

### Using Python Script

1. Run the training script:
   ```bash
   python bipedal_walker_rl.py
   ```
2. Monitor the logs in the `logs/` directory.

## Implementation Details

The agent is implemented using the **Soft Actor-Critic (SAC)** algorithm, which is an off-policy actor-critic method that maximizes a trade-off between expected return and entropy.

Key components:
- **Actor Network**: Outputs mean and log standard deviation for the Gaussian policy.
- **Critic Networks**: Two Q-value networks to mitigate overestimation bias.
- **Replay Buffer**: Stores experience tuples for off-policy learning.
- **Automatic Entropy Tuning**: Adjusts the temperature parameter $\alpha$ during training.
