import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import rl_agents
import csv

# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = TrafficEnv(rewards=rewards, max_steps=1000)

# initialize the agent
# agent = rl_planners.ValueIterationPlanner(env)
# agent = rl_planners.PolicyIterationPlanner(env)
# agent = rl_agents.QLearningAgent(env)
agent = rl_agents.SARSAgent(env)

# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}

# set light state variables
RED, GREEN = 0, 1

try:
    with open('traffic_simulation_output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Agent', 'Step', 'NS Cars', 'EW Cars', 'Light NS', 'Reward', 'Terminated', 'Truncated'])
        print("CSV file opened and header written.")
        terminated, truncated = False, False
        counter = 0

        while not terminated and not truncated:
            print(f"Step {counter}: Starting simulation step.")
            action = agent.choose_action(observation)
            print(f"Step {counter}: Action chosen - {action}")
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Step {counter}: Environment stepped. Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            ns, ew, light = tuple(observation)
            light_color = "GREEN" if light == GREEN else "RED"
            print(f"Step: {counter}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            env.render()
            time.sleep(0.8)
            writer.writerow([type(agent).__name__, counter, ns, ew, light_color, reward, terminated, truncated])
            print(f"Step {counter} written to CSV.")
            counter += 1

            # reset the environment if terminated or truncated
            if terminated or truncated:
                print("\nTERMINATED OR TRUNCATED, RESETTING...\n")
                observation, info = env.reset(), {}
                terminated, truncated = False, False
                counter = 0

    print("CSV file closed.")
except Exception as e:
    print(f"An error occurred: {e}")

# close the environment
env.render(close=True)

# while (not terminated and not truncated):
#     # use the agent's policy to choose an action
#     action = agent.choose_action(observation)
#     # step through the environment with the chosen action
#     observation, reward, terminated, truncated, info = env.step(action)

#     # unpack the state to get the number of cars and traffic light state
#     ns, ew, light = tuple(observation)
#     light_color = "GREEN" if light == GREEN else "RED"
#     # print the current state
#     print(f"Step: {counter}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
#     # render the environment at each step
#     env.render()
#     # add a delay to slow down the rendering for better visualization
#     time.sleep(0.8)

#     counter += 1

#     # reset the environment if terminated or truncated
#     if terminated or truncated:
#         print("\nTERMINATED OR TRUNCATED, RESETTING...\n")
#         observation, info = env.reset(), {}
#         terminated, truncated = False, False
#         counter = 0

# # close the environment
# env.render(close=True)
