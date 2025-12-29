# AISF Application - BipedalWalker RL Agent

This repository contains the implementation of a Reinforcement Learning agent to solve the `BipedalWalker-v3` environment as part of the AISF Application. For report reviewing, look [here](report.md)

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

The notebook automatically detects hardware and configures parallel environments:

```python
# Device auto-detection (cell 3)
if torch.backends.mps.is_available():
    device = torch.device("mps")  # macOS Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")

# Parallel environment setup (cell 15)
NUM_ENVS = 32  # Colab with CUDA: 32 | macOS with MPS: 4 | Rendering: 1
```

### Recommended Parallel Environment Counts

| Platform | Hardware | NUM_ENVS | Notes |
|----------|----------|----------|-------|
| **Google Colab** | T4 GPU (CUDA) | 32 | Maximize sample collection |
| **macOS** | Apple Silicon (MPS) | 4 | Balance speed and stability |
| **Rendering/Video** | Any | 1 | Single env for video recording |

The training loop uses `gym.vector.AsyncVectorEnv` internally to run multiple environments in parallel, significantly speeding up data collection while decorrelating experience for the replay buffer.

## Quick Start

### Notebook Workflow

1. **Open** `bipedal_walker_rl.ipynb`
2. **Install dependencies** (first two cells):
   - For Colab: `!pip install swig` and `!pip install gymnasium[box2d]`, then mount Google Drive
   - For macOS: `brew install swig`, then `pip install gymnasium[box2d]`
3. **Configure training parameters** (cell 15):
   ```python
   MAX_EPISODES = 2000
   NUM_ENVS = 32  # Use 32 on Colab (CUDA), 4 on macOS (MPS)
   LEARNING_RATE = 1e-4
   RENDER_FREQ = 0  # Set to 0 during training
   ```
4. **Choose training mode**:
   - **With Obstacles** (default): Train agent to navigate platforms, gaps, and slopes
   - **Baseline**: Comment out obstacles section and uncomment baseline section
5. **Run training** cells to train and save checkpoints
6. **Visualize results** with training curves and comparison plots
7. **Generate videos** from saved checkpoints

### Training & Evaluation Flow

- **Training**: Uses vectorized environments (32 on Colab CUDA, 4 on macOS MPS) with rendering disabled (`RENDER_FREQ=0`)
- **Checkpoints**: Saved every 10 episodes to `logs/<run_id>/checkpoint_ep<N>.pth`
- **Evaluation**: Load checkpoint and run with `ObstacleBipedalWrapper` to generate videos with obstacles visible
- **Videos**: Saved to `videos/` directory with `.mp4` format

### Video Generation from Checkpoint

```python
# Load checkpoint and generate video with obstacles
checkpoint_path = "logs/<run_id>/checkpoint_ep990.pth"
eval_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
eval_env = ObstacleBipedalWrapper(eval_env, difficulty=0.7, seed=42)

# Initialize agent and load weights
eval_agent = SACAgent(state_dim, action_dim, device=device)
checkpoint = torch.load(checkpoint_path, map_location=device)
eval_agent.actor.load_state_dict(checkpoint['actor_state_dict'])

# Generate and save video (see final cell in notebook)
```

## Troubleshooting

- **Box2D build errors**: Ensure `swig` is installed (`apt-get install -y swig` on Colab, `brew install swig` on macOS)
- **PyTorch device**: Confirm `device` resolves to `cuda` on Colab or `mps` on macOS; otherwise training will run on CPU
- **Slow or choppy rendering**: Use `NUM_ENVS=1` and avoid rendering during training (`RENDER_FREQ=0`)
- **Obstacles not visible in video**: Ensure `ObstacleBipedalWrapper` is applied when creating the evaluation environment

## Implementation Details

The notebook implements the **Soft Actor-Critic (SAC)** algorithm with the following components:

### Core Architecture
- **Actor Network**: 2-layer MLP (state_dim → 256 → 256 → action_dim) with Gaussian policy
- **Twin Critic Networks**: Two Q-networks to mitigate overestimation bias
- **Replay Buffer**: Stores 1M transitions for off-policy learning
- **Target Networks**: Soft-updated with τ=0.005 (Polyak averaging)
- **Automatic Entropy Tuning**: Learns temperature parameter α dynamically

### Obstacle Environment
- **ObstacleBipedalWrapper**: Procedurally generates platforms, gaps, and slopes on terrain
- **Configurable difficulty**: 0.0 (easy) to 1.0 (hard), default 0.7
- **Terrain modifications**: Adds ramps, pits, and stepping stones to increase challenge

### Training Configuration
- **Parallel Environments**: 32 async environments on Colab, 4 on macOS
- **Learning Rate**: 1e-4 (stable for long training)
- **Batch Size**: 256 samples per update
- **Updates per Step**: 1 gradient update per environment step
- **Start Steps**: 10,000 random exploration steps before learning
