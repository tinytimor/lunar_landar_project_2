import time
import gymnasium as gym
from robot_environment import RobotEnv
import rl_agents

# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = RobotEnv(rewards = rewards, max_steps=1000)

# initialize the agent
# agent = rl_agents.QLearningAgent(env,gamma=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.999, episodes=100)
agent = rl_agents.SARSAgent(env, episodes=200)
agent.train()

# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}

# run the environment until terminated or truncated
terminated, truncated = False, False
counter = 0
while (not terminated and not truncated):
    # use the agent's policy to choose an action
    action = agent.choose_action(observation)
    # step through the environment with the chosen action
    observation, reward, terminated, truncated, info = env.step(action)
    # print the current state
    print(f"Step: {counter}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    # render the environment at each step
    env.render()
    # Add a delay to slow down the simulation for better visualization
    time.sleep(0.1)

    counter += 1

    # reset the environment if terminated or truncated
    if terminated or truncated:
        print("\nTERMINATED OR TRUNCATED, RESETTING...\n")
        observation, info = env.reset(), {}
        terminated, truncated = False, False
        counter = 0

# close the environment
env.render(close=True)
