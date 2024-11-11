import sys
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import math  
from traffic_simulator import TrafficSim
from traffic_simulator import TrafficRenderer

# constants for traffic light actions
RED, GREEN = 0, 1


class TrafficEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, max_cars_dir=20, max_cars_total=30, lambda_ns=2, lambda_ew=3, cars_leaving=5, rewards=None, max_steps=1000): # THEY GET
        """
        Initialize the environment with specified parameters.

        Args:
            max_cars_dir (int): Maximum number of cars allowed in a single direction (north-south or east-west).
            max_cars_total (int): Maximum number of cars allowed in total across both directions.
            lambda_ns (int): Poisson rate parameter for car arrivals in the north-south direction.
            lambda_ew (int): Poisson rate parameter for car arrivals in the east-west direction.
            cars_leaving (int): Number of cars leaving the intersection per timestep.
            rewards (dict): Reward values for different traffic states.
            max_steps (int): Maximum number of steps per episode.
        """

        # set the main parameters
        self.max_cars_dir = max_cars_dir
        self.max_cars_total = max_cars_total
        self.lambda_ns = lambda_ns
        self.lambda_ew = lambda_ew

        # set the rewards function
        self.rewards = rewards

        # setting max number of steps per episode and keeping track
        self.max_steps = max_steps
        self.current_step = 0

        # two states for each direction (N/S and E/W) and one for the traffic light
        self.nS = (self.max_cars_dir + 1) ** 2 * 2  # number of states
        self.nA = 2  # number of actions (0: keep, 1: switch)

        # define action and observation spaces
        self.action_space = spaces.Discrete(self.nA)
        sizes = [self.max_cars_dir + 1, self.max_cars_dir + 1, 2]
        self.observation_space = spaces.MultiDiscrete(sizes)

        # initial state distribution
        self.isd = np.indices(sizes).reshape(len(sizes), -1).T

        # initial state
        """
        random_index = np.random.choice(self.isd.shape[0])
        self.s = tuple(self.isd[random_index]) # (ns,ew,light)
        """
        self.s = (0,0,1)

        # initialize simulator with environment parameters
        self.sim = TrafficSim(max_cars_dir, lambda_ns, lambda_ew, cars_leaving, self.s[0], self.s[1], self.s[2])
        # initialize renderer in human mode
        self.renderer = TrafficRenderer(self.sim, "human")

        # determine the transition probability matrix
        print("Building transition matrix...")
        self.P = self._build_transition_prob_matrix()
        print("Transition matrix built.")

    def _build_transition_prob_matrix(self):
        """Build the transition probability matrix."""
        P = {}
        for ns in range(self.max_cars_dir + 1):
            for ew in range(self.max_cars_dir + 1):
                for light in [RED, GREEN]:
                    state = (ns, ew, light)
                    P[state] = {action: [] for action in range(self.nA)}
                    for action in range(self.nA):
                        transitions = []
                        for appr_ns in range(8):
                            for appr_ew in range(8):
                                # determine the next state based on action
                                next_light = abs(light - action)
                                next_ns, next_ew, prob_next_state = self.sim.get_updated_wait_cars(ns, ew, next_light, appr_ns, appr_ew)
                                # get reward
                                reward = self.get_rewards(next_ns, next_ew, next_light)
                                done = self.is_terminal(next_ns, next_ew)
                                next_state = (next_ns, next_ew, next_light)
                                # collect all transitions for normalization
                                transitions += [(prob_next_state, next_state, reward, done)]
                        # normalize the probabilities to ensure they sum to 1
                        total_prob = sum([t[0] for t in transitions])
                        transitions = [(p / total_prob, s, r, d) for (p, s, r, d) in transitions]
                        # assign the normalized transitions to the state-action pair
                        P[state][action] = transitions
        return P

    def get_rewards(self, ns, ew, light):
        """
        Calculate the reward for a given state.

        Args:
            ns (int): Number of cars in the north-south direction.
            ew (int): Number of cars in the east-west direction.
            light (int): The current state of the traffic light in the north-south direction (0 for red, 1 for green).
            action (int): The action taken (0: keep, 1: switch).

        Returns:
            float: The calculated reward based on the given state.
        """
        # Initialize reward with the base state reward
        # reward = self.rewards.get("state", 0)  # Default reward if no "state" key exists

        # total_cars = ns + ew

        # # 1. Exponential penalty for total cars to discourage high traffic
        # if total_cars > 0:
        #     reward -= (2 ** (total_cars / 10)) - 1  # Exponential penalty for large traffic volumes
        # else:
        #     reward += 100  # High reward for clearing all traffic

        # # 2. Exponential penalties for individual directions
        # if ns > 0:
        #     reward -= (2 ** (ns / 10)) - 1
        # if ew > 0:
        #     reward -= (2 ** (ew / 10)) - 1

        # # 3. Severe penalty for exceeding capacity in either direction
        # if ns > self.max_cars_dir or ew > self.max_cars_dir:
        #     reward -= 1000  # Severe penalty for exceeding max capacity

        # # 4. Reward for low traffic in both directions (incentivize balance)
        # low_traffic_threshold = math.ceil(0.1 * self.max_cars_dir)
        # if ns <= low_traffic_threshold and ew <= low_traffic_threshold:
        #     reward += 20  # Reward for low traffic in both directions
        # elif ns <= low_traffic_threshold or ew <= low_traffic_threshold:
        #     reward += 10  # Smaller reward for low traffic in one direction

        # # 5. Reward for cars leaving the intersection (encouraging flow)
        # if light == GREEN:
        #     cars_crossed_ns = min(self.sim.cars_leaving, ns)
        #     reward += cars_crossed_ns * 10  # Reward for each car leaving in NS direction
        # else:
        #     cars_crossed_ew = min(self.sim.cars_leaving, ew)
        #     reward += cars_crossed_ew * 10  # Reward for each car leaving in EW direction

        # # 6. Enhanced penalty for imbalance (empty in one direction, congested in the other)
        # heavy_traffic_threshold = math.ceil(0.8 * self.max_cars_dir)
        # if (ns == 0 and ew > heavy_traffic_threshold) or (ew == 0 and ns > heavy_traffic_threshold):
        #     reward -= 3000  # Increased severe penalty for extreme imbalance

        # # 7. Reward for balanced traffic (to avoid constant toggling)
        # balance_threshold = 2  # Define what we consider "balanced"
        # if abs(ns - ew) <= balance_threshold:
        #     reward += 30  # Encourage balanced traffic flows

        # # 8. Penalize traffic spikes (discourage traffic build-up)
        # if total_cars > heavy_traffic_threshold:
        #     reward -= 800  # More aggressive penalty for allowing heavy traffic to build up

        # # 9. Diminishing returns for keeping the light green in one direction too long
        # # Note: Without adding new class attributes, we can approximate time in state using current_step
        # # For simplicity, assume the light has been in the current state for a number of steps proportional to current_step
        # # This is a rough approximation and can be adjusted based on training observations
        # light_duration_steps = self.current_step % 10  # Reset every 10 steps to simulate time in state
        # if light == GREEN:
        #     reward -= light_duration_steps * 2  # Diminish reward for keeping NS green too long
        # else:
        #     reward -= light_duration_steps * 2  # Diminish reward for keeping EW green too long

        # # 10. Severe penalties for keeping one direction red for too long when congested
        # if light == GREEN and ew > heavy_traffic_threshold:
        #     reward -= (light_duration_steps * 10)  # Penalize keeping EW red when congested
        # elif light == RED and ns > heavy_traffic_threshold:
        #     reward -= (light_duration_steps * 10)  # Penalize keeping NS red when congested

        # return reward
        # default reward for any state
        reward = self.rewards["state"]

        total_cars = ns + ew

        # reward for minimizing total cars (the fewer cars, the higher the reward)
        if total_cars == 0:
            # clearing all traffic
            reward += 100  
        else:
            # punishing algo
            if total_cars <= 5:
                reward -= total_cars * 2
            elif total_cars <= 10:
                reward -= 10 + (total_cars - 5) * 4
            elif total_cars <= 15:
                reward -= 30 + (total_cars - 10) * 6
            elif total_cars <= 20:
                reward -= 60 + (total_cars - 15) * 8
            else:
                reward -= 100 + (total_cars - 20) * 10

        # penalizing traffic jams
        if ns > self.max_cars_dir or ew > self.max_cars_dir:
            reward -= 200 

        # reward for clearing 
        if ns == 0:
            reward += 20 
        if ew == 0:
            reward += 20 

        # add penality if light did not switched to prevent buildup
        if ns > ew and light == RED:
            difference = ns - ew
            if difference >= 5:
                reward -= difference * 5 
        elif ew > ns and light == GREEN:

            difference = ew - ns
            if difference >= 5:
                reward -= difference * 5

        # reward switching the light to the direction with more cars
        if ns > ew and light == GREEN:
            reward += 10 
        elif ew > ns and light == RED:
            reward += 10

        # penalizing big for building up
        if ns >= self.max_cars_dir * 0.8:
            reward -= (ns / self.max_cars_dir) * 50 
        if ew >= self.max_cars_dir * 0.8:
            reward -= (ew / self.max_cars_dir) * 50  

        # adding reward based o the rate of cars moving through the intersection
        if light == GREEN:
            reward += min(self.sim.cars_leaving, ns) * 2 
        else:
            reward += min(self.sim.cars_leaving, ew) * 2  

        return reward


    def is_terminal(self, ns, ew):
        """
        Check if the state is terminal.

        Args:
            ns (int): Number of cars in the north-south direction.
            ew (int): Number of cars in the east-west direction.

        Returns:
            bool: True if the state is terminal (traffic jam), False otherwise.
        """
        # total_cars = ns + ew
        # # Terminal if total cars exceed the max total limit or any direction exceeds its max
        # if total_cars > self.max_cars_total or ns > self.max_cars_dir or ew > self.max_cars_dir:
        #     return True
        # return False
        return ns == 0 and ew == 0

    def is_truncated(self):
        """
        Check if the maximum number of steps has been reached.

        Returns:
            bool: True if the maximum number of steps has been reached, False otherwise.
        """
        return self.current_step >= self.max_steps

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action (int): The action to take in the environment.

        Returns:
            tuple: A tuple containing:
                - obs (np.ndarray): The new state after taking the action.
                - reward (float): The reward received for taking the action.
                - done (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated due to reaching the max number of steps.
                - info (dict): Additional information, such as the probability of the transition.
        """
        ns, ew, light = self.s

        # setting threshold to handle high traffic
        critical_threshold = self.max_cars_dir * 0.8  
        minor_imbalance_threshold = 2  

        # Ptrying to priortize clearing congested direction
        if ew >= critical_threshold:
            # swithcing to lear ew
            next_light = 0  
        elif ns >= critical_threshold:
            # switching to clear ns
            next_light = 1  
        elif abs(ns - ew) >= minor_imbalance_threshold:
            # switching for snall imbalance
            next_light = 1 if ns > ew else 0  
        else:
            next_light = light if action == 0 else 1 - light 

        # poission
        appr_ns = np.random.poisson(self.lambda_ns)
        appr_ew = np.random.poisson(self.lambda_ew)

        # ns green
        if next_light == 1:  # Green for NS
            cars_leaving_ns = min(self.sim.cars_leaving, ns)
            next_ns = ns - cars_leaving_ns + appr_ns
            next_ns = min(next_ns, self.max_cars_dir)
            next_ew = ew + appr_ew
            next_ew = min(next_ew, self.max_cars_dir)
        else:  # ew green
            cars_leaving_ew = min(self.sim.cars_leaving, ew)
            next_ew = ew - cars_leaving_ew + appr_ew
            next_ew = min(next_ew, self.max_cars_dir)
            next_ns = ns + appr_ns
            next_ns = min(next_ns, self.max_cars_dir)
        reward = self.get_rewards(next_ns, next_ew, next_light)
    # done = self.is_terminal(next_ns, next_ew)
        done = self.is_terminal(next_ns, next_ew)
        truncated = self.is_truncated()
        self.s = (next_ns, next_ew, next_light)
        self.current_step += 1

        obs = np.array(self.s)
        info = {}
        return obs, reward, done, truncated, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        # randomly select the number of cars in the NS and EW directions and the traffic light state
        random_index = np.random.choice(self.isd.shape[0])
        # set the initial state
        s = self.isd[random_index]
        self.s = tuple(s)  # (ns, ew, light)
        # reset simulator object
        self.sim.reset(*self.s)
        self.current_step = 0
        if return_info:
            return np.array(self.s), {}
        return np.array(self.s)

    def render(self, close=False):
        """Render the environment."""
        if close and self.renderer:
            if self.renderer:
                self.renderer.close()
            return

        if self.renderer:
            return self.renderer.render(*self.s)
