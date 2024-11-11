import numpy as np

class Agent:
    """
    Base RL agent class for Q-Learning and SARSA.

    Args:
        env (gym.Env): The environment to train in.
        gamma (float): Discount factor for future rewards.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay factor for epsilon.
        episodes (int): Number of training episodes.
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=1000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.q_table = np.zeros((*self.env.observation_space.nvec, self.env.action_space.n))

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Returns:
            int: The action chosen by the agent.
        """
        # Convert the state to a tuple for indexing the Q-table
        state_tuple = tuple(state)

        # Epsilon-greedy action selection
        if np.random.uniform() < self.epsilon:
            # Explore: choose a random action
            action = self.env.action_space.sample()
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            q_values = self.q_table[state_tuple]
            max_q = np.max(q_values)
            # In case of ties, randomly choose among the best actions
            actions_with_max_q = np.flatnonzero(q_values == max_q)
            action = np.random.choice(actions_with_max_q)

        return action


    def train(self):
        """
        Train the agent using the specified RL algorithm.

        Returns:
            np.ndarray: The updated Q-table after training.
        """
        previous_q_table = np.copy(self.q_table)
        initial_convergence_threshold = 1e-2  # Start with a higher threshold
        final_convergence_threshold = 1e-5    # End with a much smaller threshold for precision
        smoothing_window = 100  # Number of episodes over which to average rewards
        reward_history = []
        best_avg_reward = -np.inf  # Track best average reward for adaptive learning
        
        # Log training progress every X episodes
        log_interval = 50

        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            truncated = False
            total_episode_reward = 0

            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_episode_reward += reward  # Track total reward for this episode

            # Decay epsilon slowly
            self.epsilon = max(0.01, self.initial_epsilon * np.exp(-self.epsilon_decay * (episode / 2)))

            # Decay learning rate more slowly
            self.alpha = max(self.min_alpha, self.initial_alpha / (1 + episode / 10))  # Adjusted decay rate

            # Store the reward for smoothing
            reward_history.append(total_episode_reward)
            if len(reward_history) > smoothing_window:
                reward_history.pop(0)  # Maintain a sliding window of rewards

            # Calculate the average reward over the last N episodes
            avg_reward = np.mean(reward_history)

            # Adapt convergence threshold dynamically (reduce over time)
            convergence_threshold = initial_convergence_threshold - \
                                    (initial_convergence_threshold - final_convergence_threshold) * (episode / self.episodes)

            # Check for Q-table convergence after every 10 episodes
            if episode % 10 == 0:
                max_change = np.abs(self.q_table - previous_q_table).max()
                previous_q_table = np.copy(self.q_table)

                # Log training progress every X episodes
                if episode % log_interval == 0:
                    print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Max Q-Value Change: {max_change:.6f}, Epsilon: {self.epsilon:.4f}, Alpha: {self.alpha:.4f}")

                if max_change < convergence_threshold:
                    print(f"Converged after {episode} episodes with max Q-value change: {max_change:.6f}")
                    break

            # Update best average reward for adaptive learning rate or exploration rate
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                # Optionally slow down the decay of epsilon/alpha if performance is improving
                # self.epsilon_decay = self.epsilon_decay * 0.95
                # self.alpha = max(self.min_alpha, self.alpha * 0.9)

        return self.q_table
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table (to be implemented in subclasses).
        """
        pass


class QLearningAgent(Agent):
    """Q-Learning agent implementing the off-policy TD control."""

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-Learning algorithm.

        Args:
            state (tuple): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state of the environment.
            done (bool): Whether the episode is complete.

        Returns:
            None
        """
        state_index = tuple(state)
        next_state_index = tuple(next_state)

        # Q-learning update rule with double Q-learning
        q_next_max = np.max(self.q_table[next_state_index])
        next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.gamma * self.q_table[next_state_index + (next_action,)] * (not done)
        td_error = td_target - self.q_table[state_index + (action,)]
        self.q_table[state_index + (action,)] += self.alpha * td_error


class SARSAgent(Agent):
    """SARSA agent implementing the on-policy TD control."""

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the SARSA algorithm.

        Args:
            state (tuple): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state of the environment.
            done (bool): Whether the episode is complete.

        Returns:
            None
        """
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        # Choose next action using the current policy (epsilon-greedy)
        next_action = self.choose_action(next_state)

        # SARSA update rule
        current_q = self.q_table[state_tuple + (action,)]
        next_q = self.q_table[next_state_tuple + (next_action,)] if not done else 0
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state_tuple + (action,)] += self.alpha * td_error
