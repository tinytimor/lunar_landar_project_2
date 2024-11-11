import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time
import datetime
import os
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 128        
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 5e-3
WEIGHT_DECAY = 0

def hidden_init(layer):
    """Initialize hidden layers."""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Network."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model."""
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Network."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class OUNoise:
    """Ornstein-Uhlenbeck process for generating noise."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.size = size
        self.seed = random.seed(seed)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, action_low, action_high):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.action_low = action_low
        self.action_high = action_high

        # Actor Network (local and target)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (local and target)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn if enough samples are available
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        action = action.squeeze(0)
        return np.clip(action, self.action_low, self.action_high)

    def reset(self):
        """Reset the noise process."""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using sampled experiences."""
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # getting the expected next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # calculting Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # calculating critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # minizming the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_training_scores(scores, average_scores, total_time, save_path='ddpg/ddpg_training_progress.png'):
    """Plot training scores and average scores with the total training time."""
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Score per Episode', alpha=0.6)
    avg_episodes = [i for i in range(1, len(scores)+1)]
    plt.plot(avg_episodes, average_scores, 
             label='Average Score over 100 Episodes', linewidth=2)
    plt.axhline(y=200, color='r', linestyle='--', label='Target Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Training Progress (Total Time: {str(datetime.timedelta(seconds=int(total_time)))})')
    plt.legend()

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

def plot_evaluation_scores(eval_scores, average_score, total_time, save_path='ddpg/ddpg_evaluation_progress.png'):
    """Plot evaluation scores with the total evaluation time."""
    plt.figure(figsize=(12, 6))
    episodes = [i for i in range(1, len(eval_scores)+1)]
    plt.plot(episodes, eval_scores, label='Score per Episode', alpha=0.6)
    plt.axhline(y=average_score, color='g', linestyle='--', label=f'Average Score: {average_score:.2f}')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Score')
    plt.title(f'Evaluation Results (Total Time: {str(datetime.timedelta(seconds=int(total_time)))})')
    plt.legend()

    # Ensure directory exists before saving
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

# Training function
def train_ddpg(env, agent, n_episodes=1000, max_t=1000, target_score=200.0):
    """Train the agent using DDPG, track time, plot results, and save data to CSV."""
    scores = []  # List to store the score from each episode
    average_scores = []  # List to store the average score over 100 episodes
    scores_window = deque(maxlen=100)  # For calculating running average
    start_time = time.time()  # Start the timer

    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()  # Properly unpack the reset function
        agent.reset()  # Reset the noise process
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated:
                break

        scores_window.append(score)  # Save the latest score
        scores.append(score)  # Store score for the current episode

        # Compute average score over the last 100 episodes
        if i_episode >= 100:
            avg_score = np.mean(scores_window)
            average_scores.append(avg_score)
        else:
            average_scores.append(np.nan)

        if i_episode % 100 == 0:
            avg_last_100 = np.mean(scores_window)
            current_time = time.time() - start_time
            print(f'Episode {i_episode}/{n_episodes}   Average Score (last 100 episodes): {avg_last_100:.2f}')
            print(f"Time Elapsed: {str(datetime.timedelta(seconds=int(current_time)))}")

            checkpoint_dir = 'ddpg/checkpoints'
            ensure_dir(checkpoint_dir)
            torch.save(agent.actor_local.state_dict(), f'{checkpoint_dir}/ddpg_actor_checkpoint_{i_episode}.pth')
            torch.save(agent.critic_local.state_dict(), f'{checkpoint_dir}/ddpg_critic_checkpoint_{i_episode}.pth')

        if i_episode >= 100 and average_scores[-1] >= target_score:
            print(f'\nEnvironment solved in {i_episode} episodes! Average Score: {average_scores[-1]:.2f}')
            final_dir = 'ddpg/final_models'
            ensure_dir(final_dir)
            torch.save(agent.actor_local.state_dict(), f'{final_dir}/ddpg_actor_final.pth')
            torch.save(agent.critic_local.state_dict(), f'{final_dir}/ddpg_critic_final.pth')
            break

    total_time = time.time() - start_time 
    plot_training_scores(scores, average_scores, total_time) 
    df = pd.DataFrame({
        'Episode': [i for i in range(1, len(scores) + 1)],
        'Score': scores,
        'Average Score (last 100 episodes)': average_scores
    })

    ensure_dir('ddpg/')
    df.to_csv('ddpg/training_results.csv', index=False)

    return scores, average_scores, total_time

def evaluate_agent(env, agent, n_episodes=100, max_t=1000, render=True):
    """Evaluate the agent's performance by running episodes with rendering."""
    eval_scores = []
    start_time = time.time()

    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, add_noise=False)  
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            score += reward
            if render:
                env.render()  
                time.sleep(0.01) 
            if done or truncated:
                break
        eval_scores.append(score)
        print(f"Evaluation Episode {i_episode}: Score = {score}")

    total_time = time.time() - start_time
    average_score = np.mean(eval_scores)
    print(f"\nEvaluation over {n_episodes} episodes: Average Score = {average_score:.2f}")
    print(f"Total Evaluation Time: {str(datetime.timedelta(seconds=int(total_time)))}")

    plot_evaluation_scores(eval_scores, average_score, total_time)

    df_eval = pd.DataFrame({'Evaluation Episode': [i for i in range(1, n_episodes + 1)],'Score': eval_scores})
    ensure_dir('ddpg/')
    df_eval.to_csv('ddpg/evaluation_results.csv', index=False)

    return eval_scores, average_score, total_time

def main():
    env_name = 'LunarLanderContinuous-v2' 
    # setting the pygame enviroment to see the gane
    env = gym.make(env_name, render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    random_seed = 10
    action_low = env.action_space.low
    action_high = env.action_space.high
    agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=random_seed,
                     action_low=action_low, action_high=action_high)

    # # Train the agent
    # print("Starting Training...")
    # scores, avg_scores, total_time = train_ddpg(env, agent)
    # env.close()
    # print("Training Completed.")

    actor_path = 'ddpg/experiment1_ddpg_actor.pth'
    critic_path = 'ddpg/experiment1_ddpg_critic_final.pth'
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        agent.actor_local.load_state_dict(torch.load(actor_path))
        agent.critic_local.load_state_dict(torch.load(critic_path))
        print("Trained models loaded successfully.")
    else:
        print("Trained models not found. Please ensure that training has completed and models are saved.")
        return
    env = gym.make(env_name, render_mode='human')

    print("Starting Evaluation...")
    eval_scores, avg_eval_score, eval_time = evaluate_agent(env, agent, n_episodes=100, render=True)
    env.close()
    print("Evaluation Completed.")

if __name__ == "__main__":
    main()
