import functools
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from collections import deque
import random
import os
import glob
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Determine device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=int(1e6)):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.done = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )
    
    def __len__(self):
        return self.size

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) * 0.1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        return mu, self.log_std
    
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        
        # Log prob calculation
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        # Critic takes state and action as input
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_value(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, action_scale=1.0, device="cpu", learning_rate=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = device
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.batch_size = 256
        self.buffer_size = int(1e6)
        # Automatic entropy tuning target: -dim(A)
        self.target_entropy = -float(action_dim)
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, self.buffer_size)
        
        # Log alpha for entropy adjustment
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if state.ndim == 1:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = torch.FloatTensor(state).to(self.device)
                
            if deterministic:
                mu, _ = self.actor(state)
                action = torch.tanh(mu)
                return action.cpu().numpy() # Return batch if input was batch
            
            action, _ = self.actor.sample(state)
            return action.cpu().numpy()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actions_pred, log_pi = self.actor.sample(states)
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_pi - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_pi + 1).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class AlternatingLegsRewardWrapper(gym.Wrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale
        
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        # BipedalWalker-v3 Observation Space indices:
        # 4: hip_joint_1_angle
        # 5: hip_joint_1_speed
        # 9: hip_joint_2_angle
        # 10: hip_joint_2_speed
        
        hip1_speed = state[5]
        hip2_speed = state[10]
        
        # Reward for moving hips in opposite directions
        # If hip1_speed * hip2_speed < 0, they are moving in opposite directions (good)
        # We negate the product so that:
        # - Negative product (good) -> Positive reward
        # - Positive product (bad) -> Negative penalty
        alternating_reward = - (hip1_speed * hip2_speed)
        
        # Add to total reward
        reward += alternating_reward * self.scale
        
        return state, reward, done, truncated, info

def train_agent(env_name="BipedalWalker-v3", max_episodes=1000, max_steps=1000, device="cpu", render=False, learning_rate=3e-4, updates_per_step=1, start_steps=10000, num_envs=1, alternating_legs_scale=5.0):
    # Create environment
    if num_envs > 1:
        # Vectorized environment for faster data collection
        # Use functools.partial to pass arguments to the wrapper class
        wrapper_cls = functools.partial(AlternatingLegsRewardWrapper, scale=alternating_legs_scale)
        env = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode="async", wrappers=[wrapper_cls])
        print(f"Using {num_envs} vectorized environments with AlternatingLegsRewardWrapper (scale={alternating_legs_scale})")
    elif render:
        env = gym.make(env_name, render_mode="human")
        env = AlternatingLegsRewardWrapper(env, scale=alternating_legs_scale)
        print(f"Using AlternatingLegsRewardWrapper with scale={alternating_legs_scale}")
    else:
        env = gym.make(env_name)
        env = AlternatingLegsRewardWrapper(env, scale=alternating_legs_scale)
        print(f"Using AlternatingLegsRewardWrapper with scale={alternating_legs_scale}")
        
    if num_envs > 1:
        state_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.shape[0]
        action_scale = float(env.single_action_space.high[0])
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_scale = float(env.action_space.high[0])
    
    # Set random seeds
    set_seed(42)
    
    # Initialize agent
    print(f"Initializing SAC Agent on device: {device} with LR: {learning_rate}")
    agent = SACAgent(state_dim, action_dim, action_scale, device=device, learning_rate=learning_rate)
    
    # Training loop
    total_steps = 0
    episode_rewards = []
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Progress bar
    # For vec env, we count total steps across all envs or simple steps? 
    # Usually episodes is harder to track in vec env without wrappers.
    # We'll treat max_episodes as "update cycles" or similar if vec env, or just track differently.
    # Simplified: loop max_episodes, but for vec env this might be faster.
    pbar = tqdm(range(max_episodes), desc="Training Progress", unit="ep")
    
    current_episode = 0
    
    # Reset env
    state, _ = env.reset()
    
    # If not vec env, wrap state to be consistent (1, dim) if we want unified code, 
    # but our select_action handles it. 
    # For vec env, state is already (num_envs, dim).
    
    while current_episode < max_episodes:
        episode_reward = 0 # This won't be accurate for vec env per step
        # For vec env, we need to track rewards per env
        if num_envs > 1:
            current_rewards = np.zeros(num_envs)
            
        for step in range(max_steps):
            # Select action
            if total_steps < start_steps:
                if num_envs > 1:
                    action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
            else:
                # Agent expects single state, modify for batch
                if num_envs > 1:
                    action = agent.select_action(state, deterministic=False) # select_action needs update for batch
                else:
                    action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Handle done/truncated for buffer
            if num_envs > 1:
                done_flag = done | truncated
                current_rewards += reward
                
                # Check for completed episodes
                for i in range(num_envs):
                    if done_flag[i]:
                        # Handle terminal observation
                        if "final_observation" in info:
                            real_next_state = info["final_observation"][i]
                            # Store the transition with the real terminal state
                            # But wait, next_state[i] is already reset.
                            # We need to add (state[i], action[i], reward[i], real_next_state, True)
                            # For simple vectorized implementation, we might add them individually or construct batch
                            pass
                        
                        episode_rewards.append(current_rewards[i])
                        current_rewards[i] = 0
                        current_episode += 1
                        pbar.update(1)
                        if current_episode >= max_episodes:
                            break
                
                # For buffer addition, we need to use real_next_states where done is True
                # Copy next_state to real_next_state
                real_next_states = next_state.copy()
                if "final_observation" in info:
                    # 'final_observation' is a list of arrays for done envs or None
                    # Depending on gym version. In gymnasium, it's usually masked.
                    # info["final_observation"] is array of obs where done is True
                    # info["_final_observation"] is boolean mask
                    if "_final_observation" in info:
                        mask = info["_final_observation"]
                        for i, is_final in enumerate(mask):
                            if is_final:
                                real_next_states[i] = info["final_observation"][i]
                
                agent.replay_buffer.add(state, action, reward, real_next_states, done_flag)
                
            else:
                done_flag = done or truncated
                # Store transition
                agent.replay_buffer.add(state, action, reward, next_state, done_flag)
                episode_reward += reward
                
                if done_flag:
                    episode_rewards.append(episode_reward)
                    current_episode += 1
                    pbar.update(1)
                    state, _ = env.reset()
                    break
            
            state = next_state
            total_steps += num_envs
            
            # Update agent
            if len(agent.replay_buffer) > agent.batch_size and total_steps >= start_steps:
                for _ in range(updates_per_step * num_envs): # Scale updates with num_envs
                    agent.update()
                    
        # Logging and Checkpointing (moved outside inner loop for vec env logic compatibility)
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-10:])
            pbar.set_postfix({
                'Last Reward': f'{episode_rewards[-1]:.2f}',
                'Avg Reward (10)': f'{avg_reward:.2f}',
                'Total Steps': total_steps
            })
            
            if (current_episode) % 10 == 0:
                # Avoid spamming log
                pass
                
            if (current_episode) % 50 == 0:
                 os.makedirs("logs", exist_ok=True)
                 torch.save({
                     'actor_state_dict': agent.actor.state_dict(),
                     'critic1_state_dict': agent.critic1.state_dict(),
                     'critic2_state_dict': agent.critic2.state_dict(),
                     'episode': current_episode,
                     'learning_rate': learning_rate
                 }, f"logs/bipedal_walker_checkpoint_ep{current_episode}.pth")

    env.close()
    return episode_rewards, agent

def record_video(agent, env_name="BipedalWalker-v3", filename="bipedal_walker", device="cpu"):
    # Create environment with render mode
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Wrap environment to record video
    # We force record the first episode
    video_folder = "videos"
    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(env, video_folder=video_folder, name_prefix=filename, episode_trigger=lambda x: True)
    
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated):
        # Use deterministic policy for evaluation
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
    
    env.close()
    print(f"Evaluation Run - Total Reward: {total_reward:.2f}")
    
    # Find the video file
    mp4_files = glob.glob(f"{video_folder}/{filename}-episode-0.mp4")
    if mp4_files:
        print(f"Video saved to {mp4_files[0]}")
        return mp4_files[0]
    return None

def run_ablation_study(device="cpu"):
    # Compare different learning rates
    learning_rates = [1e-3, 3e-4, 1e-4]
    results = {}
    
    print("Starting Ablation Study on Learning Rates...")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    for lr in learning_rates:
        print(f"\nTesting Learning Rate: {lr}")
        # Run training for fewer episodes for ablation study to save time, or full duration if desired
        # Using 500 episodes for ablation study demonstration
        ablation_episodes = 500 
        rewards, _ = train_agent(max_episodes=ablation_episodes, device=device, learning_rate=lr)
        results[f"lr_{lr}"] = rewards
        
        # Save intermediate result
        os.makedirs("logs", exist_ok=True)
        np.save(f"logs/ablation_rewards_lr_{lr}.npy", rewards)
    
    print("Ablation study completed. Plotting results...")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for name, rewards in results.items():
        # Smooth rewards for better visualization
        window = 10
        smoothed_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        plt.plot(smoothed_rewards, label=name)
        
    plt.title("Ablation Study: Learning Rates (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    plot_path = f"logs/ablation_study_lr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path)
    print(f"Ablation plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC Agent on BipedalWalker-v3")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--updates", type=int, default=2, help="Number of gradient updates per environment step")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments to run")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "ablation"], help="Mode to run: train or ablation")
    parser.add_argument("--no-log-file", action="store_true", help="Disable logging to file")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Redirect output to log file unless disabled
    if not args.no_log_file:
        import sys
        # Keep original stdout for tqdm to work properly if needed, but here we redirect everything
        # For better tqdm support with file logging, one might use TQDM's write or separate logging
        # But matching previous behavior:
        sys.stdout = open(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", 'w')
    
    # Determine device
    if args.device:
        target_device = args.device
    else:
        target_device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {target_device}")
    
    if args.mode == "train":
        # Start training
        print(f"Starting BipedalWalker training on {target_device} with {args.episodes} episodes, {args.updates} updates per step, and {args.num_envs} envs...")
        rewards, trained_agent = train_agent(max_episodes=args.episodes, device=target_device, render=args.render, updates_per_step=args.updates, num_envs=args.num_envs)
        print("Training completed!")
        
        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plot_path = f"logs/training_rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        print(f"Training plot saved to {plot_path}")
        plt.close()

        # Record video
        print("Recording evaluation video...")
        record_video(trained_agent, device=target_device)
        
    elif args.mode == "ablation":
        run_ablation_study(device=target_device)
