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

# ============================ 1. Hyperparameter Definitions ============================

# Base Hyperparameters
BASE_LEARNING_RATE_ACTOR = 1e-4
BASE_LEARNING_RATE_CRITIC = 5e-3
BASE_BUFFER_SIZE = int(1e6)  
BASE_BATCH_SIZE = 128        
BASE_GAMMA = 0.99
BASE_TAU = 1e-3
BASE_WEIGHT_DECAY = 0

HYPERPARAMETERS_TO_TUNE = {
    'learning_rate_critic': [1e-3, 5e-3, 1e-2],
    'batch_size': [64, 128, 256],
    'gamma': [0.985, 0.99, 0.995],
    'tau': [1e-2, 1e-3, 1e-4]
}

# ============================ 2. Neural Network Architectures ============================

def hidden_init(layer):
    """Initialize hidden layers."""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Actor Network for continuous actions
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

# Critic Network
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

# ============================ 3. Noise Process and Replay Buffer ============================

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

# ============================ 4. DDPG Agent Implementation ============================

class DDPGAgent:
    def __init__(self, state_size, action_size, random_seed, hyperparameters, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # hyperparameters
        self.gamma = hyperparameters['gamma']
        self.tau = hyperparameters['tau']
        self.batch_size = hyperparameters['batch_size']
        self.learning_rate_actor = hyperparameters['learning_rate_actor']
        self.learning_rate_critic = hyperparameters['learning_rate_critic']
        self.weight_decay = hyperparameters['weight_decay']

        # action bounds
        self.action_low = action_low
        self.action_high = action_high

        # actor netwrok (local and target)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)

        # critic network (local and target)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.learning_rate_critic, weight_decay=self.weight_decay)
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        self.noise = OUNoise(action_size, random_seed)

        self.memory = ReplayBuffer(action_size, hyperparameters['buffer_size'], 
                                   hyperparameters['batch_size'], random_seed)

    def hard_update(self, target, source):
        """copys network parameters from source to target."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state, action, reward, next_state, done):
        """saves the experience and learn."""
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise=True):
        """returns actions for given state as per current policy."""
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
        """reseting the the noise process."""
        self.noise.reset()

    def learn(self, experiences):
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
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# ============================ 5. Experiment Tracking ============================

class ExperimentTracker:
    """Handles experiment tracking by managing directories and file paths."""

    def __init__(self, base_dir='ddpg_experiments'):
        self.base_dir = base_dir
        self.experiment_count = self._get_next_experiment_number()
        self.current_experiment_dir = os.path.join(base_dir, f'experiment_{self.experiment_count}')
        os.makedirs(self.current_experiment_dir, exist_ok=True)
        
    def _get_next_experiment_number(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            return 1
        existing_experiments = [d for d in os.listdir(self.base_dir) 
                               if os.path.isdir(os.path.join(self.base_dir, d))]
        if not existing_experiments:
            return 1
        experiment_numbers = [int(exp.split('_')[1]) for exp in existing_experiments if exp.split('_')[1].isdigit()]
        return max(experiment_numbers) + 1 if experiment_numbers else 1
    
    def get_path(self, filename):
        """Returns the full path for a given filename in the current experiment directory."""
        return os.path.join(self.current_experiment_dir, filename)

# ============================ 6. Plotting Function ============================

def plot_experiment_results(scores, average_scores, hyperparameter_name, hyperparameter_value, 
                           total_time, tracker):
    """Plot scores and average scores with the total training time."""
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Score per Episode', alpha=0.6)
    avg_episodes = [i for i in range(1, len(scores)+1)]
    plt.plot(avg_episodes, average_scores, label='Average Score over 100 Episodes', linewidth=2)
    plt.axhline(y=200, color='r', linestyle='--', label='Target Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Training Progress - {hyperparameter_name}: {hyperparameter_value}\n'
              f'(Total Time: {str(datetime.timedelta(seconds=int(total_time)))})')
    plt.legend()
    
    plt.savefig(tracker.get_path(f'{hyperparameter_name}_{hyperparameter_value}_progress.png'))
    plt.close()

# ============================ 7. Training Function with Tracking ============================

def train_ddpg_with_tracking(env, agent, hyperparameter_name, hyperparameter_value, 
                             tracker, n_episodes=800, max_t=1000, target_score=200.0):
    """Train the agent using DDPG, track time, plot results, and save data to CSV."""
    scores = []  # list for the score from each episode
    average_scores = []  # list for the average score over 100 episodes
    scores_window = deque(maxlen=100)  #running the values
    start_time = time.time()

    for i_episode in range(1, n_episodes+1):
        state, info = env.reset() 
        agent.reset() 
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated:
                break

        scores_window.append(score)
        scores.append(score)
        if i_episode >= 100:
            avg_score = np.mean(scores_window)
            average_scores.append(avg_score)
        else:
            average_scores.append(np.nan)

        if i_episode % 100 == 0:
            avg_last_100 = np.mean(scores_window)
            current_time = time.time() - start_time
            print(f'Episode {i_episode}/{n_episodes}   '
                  f'Average Score (last 100 episodes): {avg_last_100:.2f}')
            print(f"Time Elapsed: {str(datetime.timedelta(seconds=int(current_time)))}")

            # save tge training model every 100 episodes
            torch.save(agent.actor_local.state_dict(), 
                       tracker.get_path(f'ddpg_actor_checkpoint_{i_episode}.pth'))
            torch.save(agent.critic_local.state_dict(), 
                       tracker.get_path(f'ddpg_critic_checkpoint_{i_episode}.pth'))

        if i_episode >= 100 and np.mean(scores_window) >= target_score:
            print(f'\nEnvironment solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.2f}')
            torch.save(agent.actor_local.state_dict(), tracker.get_path('ddpg_actor_final.pth'))
            torch.save(agent.critic_local.state_dict(), tracker.get_path('ddpg_critic_final.pth'))
            break

    total_time = time.time() - start_time  # Calculate total time taken

    plot_experiment_results(scores, average_scores, hyperparameter_name, 
                           hyperparameter_value, total_time, tracker)

    df = pd.DataFrame({
        'Episode': range(1, len(scores) + 1),
        'Score': scores,
        'Average Score (last 100 episodes)': average_scores,
        hyperparameter_name: [hyperparameter_value] * len(scores)
    })
    df.to_csv(tracker.get_path(f'{hyperparameter_name}_{hyperparameter_value}_results.csv'), 
              index=False)

    return scores, average_scores, total_time

# ============================ 8. Hyperparameter Experimentation Function ============================

def run_hyperparameter_experiment(hyperparameter_name, values_to_test):
    """Runs experiments for a specific hyperparameter across different values."""
    tracker = ExperimentTracker()
    
    print(f"\nRunning experiment for {hyperparameter_name}")
    print(f"Values to test: {values_to_test}")
    
    all_results = []
    
    for value in values_to_test:
        print(f"\nTesting {hyperparameter_name}: {value}")
        
        # making enviroment
        env = gym.make('LunarLanderContinuous-v2')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        action_low = env.action_space.low
        action_high = env.action_space.high
        
        hyperparameters = {
            'learning_rate_actor': BASE_LEARNING_RATE_ACTOR,
            'learning_rate_critic': BASE_LEARNING_RATE_CRITIC,
            'buffer_size': BASE_BUFFER_SIZE,
            'batch_size': BASE_BATCH_SIZE,
            'gamma': BASE_GAMMA,
            'tau': BASE_TAU,
            'weight_decay': BASE_WEIGHT_DECAY
        }
        
        hyperparameters[hyperparameter_name] = value
        
        agent = DDPGAgent(state_size, action_size, random_seed=10, 
                         hyperparameters=hyperparameters,
                         action_low=action_low,
                         action_high=action_high)
        
        scores, avg_scores, training_time = train_ddpg_with_tracking(
            env, agent, hyperparameter_name, value, tracker)
        
        all_results.append({
            'value': value,
            'scores': scores,
            'avg_scores': avg_scores,
            'training_time': training_time
        })
        
        env.close()
    
    plt.figure(figsize=(15, 8))
    for result in all_results:
        plt.plot([i for i in range(1, len(result['avg_scores'])+1)],
                 result['avg_scores'],
                 label=f'{hyperparameter_name}={result["value"]}')
    
    plt.axhline(y=200, color='r', linestyle='--', label='Target Score')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title(f'Comparison of Different {hyperparameter_name} Values')
    plt.legend()
    plt.savefig(tracker.get_path(f'{hyperparameter_name}_comparison.png'))
    plt.close()
    
    # making master CSV
    combined_data = []
    for result in all_results:
        for episode, (score, avg_score) in enumerate(zip(result['scores'], 
                                                        result['avg_scores']), 1):
            combined_data.append({
                'Episode': episode,
                'Score': score,
                'Average Score': avg_score,
                hyperparameter_name: result['value']
            })
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(tracker.get_path(f'{hyperparameter_name}_combined_results.csv'), 
                       index=False)

# ============================ 9. Main Execution Block ============================

if __name__ == "__main__":
    
    # # Experiment 1: Learning Rate Critic
    learning_rate_critics = [1e-3, 5e-3, 1e-4]
    run_hyperparameter_experiment('learning_rate_critic', learning_rate_critics)
    
    # # Experiment 2: Tau
    # taus = [1e-2, 1e-3, 1e-4]
    # run_hyperparameter_experiment('tau', taus)

    # Experiment 3: Gamma
    # gammas = [0.985, 0.99, 0.995]
    # run_hyperparameter_experiment('gamma', gammas)