import gym
import numpy as np
import torch
import torch.nn as nn
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair, AgentFromPolicy, NNPolicy
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from plotting import plot_metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### Network Architectures ###
class SharedActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(SharedActor, self).__init__()
        self.shared_fc = nn.Linear(input_dim, hidden_dim)
        self.shared_ln = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.actor_fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        shared = self.relu(self.shared_ln(self.shared_fc(x)))
        pi = self.softmax(self.actor_fc(shared))
        return pi

class VDNCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(VDNCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.fc_v0 = nn.Linear(hidden_dim, 1)
        self.fc_v1 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        v0 = self.fc_v0(x)
        v1 = self.fc_v1(x)
        total_v = v0 + v1
        return total_v, v0, v1

class StudentPolicy(NNPolicy):
    def __init__(self, shared_actor, base_env):
        super(StudentPolicy, self).__init__()
        self.shared_actor = shared_actor
        self.base_env = base_env

    def state_policy(self, state, agent_index):
        featurized_state = self.base_env.featurize_state_mdp(state)
        input_state = torch.FloatTensor(featurized_state[agent_index]).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.shared_actor(input_state).cpu().numpy()[0]
        return action_probs

    def act(self, state, agent_index):
        action_probs = self.state_policy(state, agent_index)
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, {"action_probs": action_probs.tolist()}

class StudentAgent(AgentFromPolicy):
    def __init__(self, policy):
        super(StudentAgent, self).__init__(policy)

    def act(self, state, agent_index):
        return self.policy.act(state, agent_index)

def load_checkpoint(checkpoint_path, obs_dim, n_actions):
    shared_actor = SharedActor(obs_dim, n_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    shared_actor.load_state_dict(checkpoint['shared_actor_state_dict'])
    shared_actor.eval()
    return shared_actor

output_dirs = ['plots', 'evaluation_results']
for dir_name in output_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

### Evaluation Parameters ###
curriculum_layouts = [
    "cramped_room",
    "asymmetric_advantages",
    "forced_coordination"
]

reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5
}

horizon = 400
episodes_per_layout = 100

# going throuhg all checkppoints
checkpoint_dir = "checkpoints"
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])

### Evaluation Loop ###
for checkpoint_file in checkpoints:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    print(f"\nEvaluating checkpoint: {checkpoint_file}")
    
    #initializing results list for this checkpoint
    checkpoint_results = []
    
    for layout in curriculum_layouts:
        print(f"Evaluating layout: {layout}")
        
        #creating the environment
        mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
        
        # getting the dimensions
        obs = env.reset()
        obs_dim = obs["both_agent_obs"][0].shape[0]
        n_actions = env.action_space.n
        
        # laoding the checkpoints
        shared_actor = load_checkpoint(checkpoint_path, obs_dim, n_actions)
        
        # creating policies
        policy = StudentPolicy(shared_actor, base_env)
        agent_pair = AgentPair(StudentAgent(policy), StudentAgent(policy))
        layout_results = []
        for episode in range(episodes_per_layout):
            obs = env.reset()
            done = False
            episode_reward = 0
            soups_made = 0
            
            while not done:
                # getting current state
                state = env.base_env.state
                
                # getting actions for both agents
                action0, _ = policy.act(state, 0)
                action1, _ = policy.act(state, 1)
                
                #taking the next step
                obs_, reward, done, info = env.step([action0, action1])
                
                # tracking the rewards and soups cooked/delivered
                episode_reward += reward
                soups_made += int(reward / 20)
                
                obs = obs_
            
            layout_results.append({
                'checkpoint': checkpoint_file,
                'layout': layout,
                'episode': episode,
                'reward': episode_reward,
                'soups_made': soups_made
            })
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes_per_layout}")
        
        checkpoint_results.extend(layout_results)
        
        # calculating the store layout averages
        avg_reward = np.mean([r['reward'] for r in layout_results])
        avg_soups = np.mean([r['soups_made'] for r in layout_results])
        print(f"Layout {layout} - Avg Reward: {avg_reward:.2f}, Avg Soups: {avg_soups:.2f}")

    # Create DataFrame for this checkpoint
    checkpoint_df = pd.DataFrame(checkpoint_results)
    
    # saving thecheckpoint-specific results
    checkpoint_name = checkpoint_file.replace('.pt', '')
    csv_path = os.path.join('evaluation_results', f'{checkpoint_name}_results.csv')
    pkl_path = os.path.join('evaluation_results', f'{checkpoint_name}_results.pkl')
    
    checkpoint_df.to_csv(csv_path, index=False)
    checkpoint_df.to_pickle(pkl_path)
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Performance for {checkpoint_name}', y=1.02, fontsize=16)
    
    # soups made and delivered plot
    plt.subplot(2, 1, 1)
    for layout in curriculum_layouts:
        layout_data = checkpoint_df[checkpoint_df['layout'] == layout]
        plt.plot(layout_data['episode'], layout_data['soups_made'], label=layout, marker='o', markersize=4)
    plt.xlabel('Episode')
    plt.ylabel('Soups Made')
    plt.legend()
    plt.title('Soups Made per Episode')
    plt.grid(True, alpha=0.3)
    
    # reward plotting
    plt.subplot(2, 1, 2)
    for layout in curriculum_layouts:
        layout_data = checkpoint_df[checkpoint_df['layout'] == layout]
        plt.plot(layout_data['episode'], layout_data['reward'], label=layout, marker='o', markersize=4)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Reward per Episode')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # saving the plots
    plot_path = os.path.join('plots', f'{checkpoint_name}_performance.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved results for {checkpoint_name}:")
    print(f"- CSV: {csv_path}")
    print(f"- DataFrame: {pkl_path}")
    print(f"- Plot: {plot_path}")

plt.figure(figsize=(15, 10))
plt.suptitle('Performance Across All Checkpoints', y=1.02, fontsize=16)

plt.subplot(2, 1, 1)
for layout in curriculum_layouts:
    avg_soups_by_checkpoint = []
    checkpoint_numbers = []
    
    for checkpoint_file in checkpoints:
        checkpoint_name = checkpoint_file.replace('.pt', '')
        df = pd.read_csv(os.path.join('evaluation_results', f'{checkpoint_name}_results.csv'))
        layout_data = df[df['layout'] == layout]
        avg_soups = layout_data['soups_made'].mean()
        avg_soups_by_checkpoint.append(avg_soups)
        checkpoint_numbers.append(int(checkpoint_name.split('_')[2]))  # Assuming format "checkpoint_cycle_X"
    
    plt.plot(checkpoint_numbers, avg_soups_by_checkpoint, label=layout, marker='o')

plt.xlabel('Checkpoint Cycle')
plt.ylabel('Average Soups Made')
plt.legend()
plt.title('Average Soups Made Across Checkpoints')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
for layout in curriculum_layouts:
    avg_reward_by_checkpoint = []
    checkpoint_numbers = []
    
    for checkpoint_file in checkpoints:
        checkpoint_name = checkpoint_file.replace('.pt', '')
        df = pd.read_csv(os.path.join('evaluation_results', f'{checkpoint_name}_results.csv'))
        layout_data = df[df['layout'] == layout]
        avg_reward = layout_data['reward'].mean()
        avg_reward_by_checkpoint.append(avg_reward)
        checkpoint_numbers.append(int(checkpoint_name.split('_')[2]))
    
    plt.plot(checkpoint_numbers, avg_reward_by_checkpoint, label=layout, marker='o')

plt.xlabel('Checkpoint Cycle')
plt.ylabel('Average Reward')
plt.legend()
plt.title('Average Reward Across Checkpoints')
plt.grid(True, alpha=0.3)

plt.tight_layout()
summary_plot_path = os.path.join('plots', 'checkpoint_comparison_summary.png')
plt.savefig(summary_plot_path, bbox_inches='tight', dpi=300)
plt.close()
summary_stats = []
for checkpoint_file in checkpoints:
    checkpoint_name = checkpoint_file.replace('.pt', '')
    df = pd.read_csv(os.path.join('evaluation_results', f'{checkpoint_name}_results.csv'))
    
    for layout in curriculum_layouts:
        layout_data = df[df['layout'] == layout]
        stats = {
            'checkpoint': checkpoint_name,
            'layout': layout,
            'avg_soups': layout_data['soups_made'].mean(),
            'std_soups': layout_data['soups_made'].std(),
            'avg_reward': layout_data['reward'].mean(),
            'std_reward': layout_data['reward'].std(),
            'max_soups': layout_data['soups_made'].max(),
            'min_soups': layout_data['soups_made'].min()
        }
        summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(os.path.join('evaluation_results', 'summary_statistics.csv'), index=False)
summary_df.to_pickle(os.path.join('evaluation_results', 'summary_statistics.pkl'))

print("\nEvaluation complete!")
print(f"Results saved in '{os.path.abspath('evaluation_results')}' directory")
print(f"Plots saved in '{os.path.abspath('plots')}' directory")


data_dir = 'training_data'

file_paths = list(Path(data_dir).glob('*.csv'))

dataframes = [pd.read_csv(file) for file in file_paths]
all_data = pd.concat(dataframes, ignore_index=True)

# creating a new columns that groups episodes into bins of 3000 for layout-wise averaging
all_data['episode_bin'] = (all_data['episode'] // 3000) * 3000

#calculatong the mean of metrics for each layout and 3000-episode bin
layout_episode_avg = all_data.groupby(['layout', 'episode_bin']).mean().reset_index()

plot_metrics(
    data=all_data, 
    layout_avg_data=layout_episode_avg,
    x_col='episode', 
    y_col='total_onion_pickups', 
    color='orange', 
    sma_color='darkorange', 
    label='Total Onion Pickups', 
    sma_label='Onion Pickups (3000-episode SMA)',
    save_path='total_onion_pickups_plot.png'
)

plot_metrics(
    data=all_data, 
    layout_avg_data=layout_episode_avg,
    x_col='episode', 
    y_col='total_dish_drops', 
    color='skyblue', 
    sma_color='red', 
    label='Total Dish Drops', 
    sma_label='Dish Drops (3000-episode SMA)',
    save_path='total_dish_drops_plot.png'
)
