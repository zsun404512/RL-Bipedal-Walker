# AISF Application - BipedalWalker RL Agent

This repository contains the implementation of a Reinforcement Learning agent to solve the `BipedalWalker-v3` environment as part of the AISF Application.

## Project Structure

- `bipedal_walker_rl.ipynb`: The main Jupyter Notebook containing the implementation, training loop, evaluation, and writeup.
- `bipedal_walker_rl.py`: A Python script version of the implementation for command-line execution.
- `logs/`: Directory where training logs are stored.
- `videos/`: Directory where evaluation videos are saved.
- `*.pth`: Model checkpoints saved during training.

## Requirements

To run this code, you need the following dependencies:

```bash
pip install gymnasium[box2d] torch numpy matplotlib ipython moviepy tqdm
```

*Note: You may need to install `swig` to build the Box2D dependencies.*

## Hardware Acceleration

The code automatically detects and uses the available hardware acceleration:
- **macOS (Apple Silicon)**: Uses Metal Performance Shaders (`mps`) for GPU acceleration.
- **NVIDIA GPUs**: Uses CUDA (`cuda`) if available.
- **CPU**: Fallback if no GPU is detected.

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

## Results

(To be filled after training)
- **Best Reward**: [Insert Best Reward]
- **Convergence Time**: [Insert Number of Episodes/Steps]

## Ablation Studies

The notebook includes a section for running ablation studies to compare different hyperparameters (e.g., learning rates).

## References

- Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." arXiv preprint arXiv:1801.01290 (2018).
