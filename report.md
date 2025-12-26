# AISF Application Report - BipedalWalker RL Agent

## Applicant Information
- **Name**: [Your Name]
- **Email**: [Your Email]

## Prior Experience
*(Please fill in your specific prior experience here. Example: I have some experience with Python and basic machine learning concepts, but this is my first deep dive into continuous control Reinforcement Learning and the Gymnasium environment.)*

## Time Breakdown
**Total Time Spent**: ~11 hours

- **Research & Reading (3 hours)**: 
  - Spent time reading the [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) documentation to understand the fundamentals of Reinforcement Learning (RL).
  - Studied the `Gymnasium` environment API and the specific dynamics of `BipedalWalker-v3`.
  - Researched advanced RL tricks and "bipedal locomotion" strategies.
  
- **Coding, Debugging & Tuning (8 hours)**:
  - Implemented the Soft Actor-Critic (SAC) algorithm from scratch.
  - Debugged tensor shape mismatches and device placement issues (CUDA/MPS).
  - Implemented parallel environments using `gym.vector` to speed up data collection.
  - Tuned reward functions to fix "shaking" behaviors and encourage stable walking.
  - Ran parallel experiments on Google Colab Pro.

## Compute Resources
The final agent was trained using **Google Colab Pro**:
- **GPU**: NVIDIA T4 Tensor Core GPU.
- **CPU**: High-RAM runtime (utilized for parallel environment stepping).
- **Parallelization**: Utilized 32 parallel environments (`num_envs=32`) to maximize sample efficiency and training speed.

## Techniques Used

### 1. Soft Actor-Critic (SAC)
I chose **Soft Actor-Critic (SAC)**, an off-policy actor-critic algorithm that maximizes a trade-off between expected return and entropy. 
- **Justification**: `BipedalWalker-v3` has a continuous action space. SAC is known for its sample efficiency and stability compared to PPO or DDPG in such tasks. The entropy maximization term encourages exploration, preventing the agent from getting stuck in local optima (like falling immediately or just vibrating).
- **Key Components**:
  - **Twin Critics (Double Q-Learning)**: Two Q-networks (`critic1`, `critic2`) are used, and the minimum Q-value is taken during updates to mitigate the positive bias in policy improvement.
  - **Automatic Entropy Tuning**: The temperature parameter $\alpha$ is automatically adjusted during training to maintain a target entropy, balancing exploration and exploitation dynamically.
  - **Polyack Averaging (Soft Updates)**: Target networks are updated slowly ($\tau=0.005$) to stabilize learning.

### 2. Reward Engineering
To overcome the difficult initial dynamics of the walker, I implemented a custom `WalkingRewardWrapper`. The default reward was insufficient for rapid convergence, often leading to suboptimal local minima.
- **Scissoring Reward**: Explicitly rewarded alternating leg movements (negative product of hip joint speeds) *only* when forward velocity was positive. This was crucial to prevent the agent from learning a "vibrating" stationary strategy.
- **Stability Penalty**: Added negative rewards for significant hull angle deviations to keep the robot upright.
- **Energy Penalty**: Penalized large actions to encourage smooth, efficient gait.

### 3. Vectorized Environments (Parallelization)
I utilized `gym.make_vec` (Async Vector Environment) to run **32 environments in parallel**. 
- **Benefit**: This drastically reduced wall-clock training time. Instead of collecting 1 step per cycle, the agent collects 32 steps. This decorrelates the experience in the replay buffer and provides a more diverse set of states for the learner, stabilizing the gradient updates.

## Discussion of Issues Encountered

Throughout the development process, I encountered several technical hurdles (documented in git history):

1.  **Reward Hacking / Local Minima**:
    - *Issue*: Initially, the agent learned to simply vibrate its legs rapidly without moving forward to maximize the "scissoring" reward component I introduced, or it would perform a split and simply stay safe.
    - *Solution*: I modified the reward function to strictly condition the "scissoring" bonus on having a positive forward velocity (`velocity > 0.05`). This forced the agent to actually move to get the bonus.

2.  **Training Instability & Divergence**:
    - *Issue*: In some runs, the critic loss would explode, or the policy would collapse after a certain number of episodes (periodic divergence).
    - *Solution*: I adjusted the learning rate and reduced the update frequency (`updates_per_step`). I also ensured that the input states were properly normalized in the network or handled via batch normalization layers implicitly by the robust SAC formulation.

3.  **Vectorization & Rendering**:
    - *Issue*: Implementing `gym.vector` caused issues with the video recording wrapper and some shape mismatches in the replay buffer (handling `(num_envs, obs_dim)` vs `(obs_dim,)`).
    - *Solution*: I refactored the `ReplayBuffer` and `Action Selection` logic to handle batched inputs. I also added a check to only render the first environment in the vectorized stack (`frames[0]`) to generate evaluation videos without crashing the training loop.

4.  **Device Management**:
    - *Issue*: Ensuring tensors were correctly moved between CPU (for environment stepping) and GPU (for gradient updates) was tricky, especially with the vectorized environments returning numpy arrays.
    - *Solution*: I implemented a robust `to_tensor` helper in the Replay Buffer and ensured explicit device casting `to(device)` before network passes.

## Ablation Studies
*Note: A full hyperparameter ablation study was planned to investigate the impact of learning rates (1e-3 vs 3e-4 vs 1e-4). However, due to the complexity of finding a stable converging algorithm first, a satisfactory baseline for rigorous comparative ablation was not finalized in time for this report. The code infrastructure for running these ablations (loops over LRs, saving results to `npy` files) is implemented in the codebase.*

## References
1.  **Soft Actor-Critic**: Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *arXiv preprint arXiv:1801.01290* (2018).
2.  **OpenAI Spinning Up**: Achiam, J. "Spinning Up in Deep RL." [spinningup.openai.com](https://spinningup.openai.com).
3.  **Gymnasium Documentation**: [gymnasium.farama.org](https://gymnasium.farama.org).
