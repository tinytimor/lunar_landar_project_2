import sys
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from robot_simulator import RobotSim
from robot_simulator import RobotRenderer


class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, n_rows=10, n_cols=10, warehouse_workers=None, warehouse_equipment=None, start = (0,0), goal=(8, 2), rewards=None, prob_success=1.0, max_steps=1000, renderer=None):
        """
        Initializes the RobotEnv environment.

        Parameters:
            n_rows (int, optional): Number of rows in the warehouse grid. Default is 10.
            n_cols (int, optional): Number of columns in the warehouse grid. Default is 10.
            warehouse_workers (list of tuples, optional): Initial positions of the warehouse workers. Default is [(2, 6), (7, 6)].
            warehouse_equipment (list of tuples, optional): Positions of the warehouse equipment. Default is [(4, 2), (4, 3), (4, 4), (4, 5)].
            start (tuple, optional): Starting position of the robot. Default is (0, 0).
            goal (tuple, optional): The goal position where the robot should deliver boxes. Default is (8, 2).
            rewards (dict): Dictionary defining the reward values for different events. Default is None.
            prob_success (float, optional): Probability of successfully executing the chosen action. Default is 1.0.
            max_steps (int, optional): Maximum number of steps allowed per episode. Default is 10000.
            renderer (RobotRenderer, optional): The renderer used for visualizing the environment. Default is None.
        """
        # set the warehouse layout
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.warehouse_workers = warehouse_workers or [(2, 6), (7, 6)]
        self.warehouse_equipment = warehouse_equipment or [(4, 2), (4, 3), (4, 4), (4, 5)]
        self.start = start
        self.goal = goal
        # set the rewards structure
        self.rewards = rewards
        # setting max number of steps per episode and tracking steps
        self.max_steps = max_steps
        self.current_step = 0
        self.robot_position = (0, 0)
        # define action and observation spaces
        self.action_space = spaces.Discrete(4) # up, right, left, down
        # total number of positions in the grid
        n_positions = n_rows * n_cols
        
        # define the observation space
        self.observation_space = spaces.Dict({
            "robot_position": spaces.Discrete(n_positions),
            "bumped_status": spaces.Discrete(5),  # 5 possible bump statuses ("", "wall", "equipment", "worker", "target")
            "curr_target": spaces.Discrete(n_positions),
        })
        # define transition probabilities
        self.prob_success = prob_success
        self.prob_fail = (1.0 - self.prob_success) / (self.action_space.n - 1)

        # initialize simulator with environment parameters
        self.sim = RobotSim(n_rows, n_cols, self.warehouse_workers, self.warehouse_equipment, self.goal, self.start)

        # initialize the state
        self.s = self.sim.get_world_state()

        # initialize renderer
        self.renderer = None

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """
        Resets the environment to the initial state.

        Parameters:
            seed (int, optional): Random seed for reproducibility. Default is None.
            return_info (bool, optional): Whether to return additional information. Default is False.
            options (dict, optional): Additional options for resetting the environment. Default is None.

        Returns:
            np.ndarray: The initial observation representing the robot's starting position.
            dict (optional): Additional information if return_info is True.
        """
        # reset the simulation state
        self.sim.reset(self.n_rows, self.n_cols, self.warehouse_workers, self.warehouse_equipment, self.goal, self.start)
        self.current_step = 0
        self.renderer = RobotRenderer(self.sim)
        self.s = self.sim.get_world_state()

        if return_info:
            return self.encode_observation(self.s), {}
        return self.encode_observation(self.s)

    def sample_action(self, action):
        """
        Sample an action based on the given action and the environment's transition probabilities.

        Parameters:
            action (int): The action chosen by the agent, where 0 = up, 1 = right, 2 = down, and 3 = left.

        Returns:
            tuple: A tuple containing the probability and the action actually taken.
        """
        if np.random.rand() < self.prob_success:
            return self.prob_success, action
        else:
            other_actions = [a for a in range(self.action_space.n) if a != action]
            chosen_action = np.random.choice(other_actions)
            return self.prob_fail, chosen_action

    def step(self, action):
        """
        Takes a step in the environment based on the action.

        Parameters:
            action (int): The action to take, where 0 = up, 1 = right, 2 = down, and 3 = left.

        Returns:
            np.ndarray: The new observation after taking the action.
            float: The reward obtained after taking the action.
            bool: Whether the episode has ended due to a terminal state.
            bool: Whether the episode has ended due to truncation (max steps reached).
            dict: Additional information such as the probability of the action being taken.
        """
        self.current_step += 1
        print('ROBOT POSITION, ', self.sim.get_world_state())
        row, col = self.sim.get_world_state()[:2]
        # moving up
        if action == 0:  
            row = max(0, row - 1)
        elif action == 1: 
        #moving down
            row = min(self.n_rows - 1, row + 1)
        elif action == 2: 
        # moving left
            col = max(0, col - 1)
        elif action == 3: 
            # moving right
            col = min(self.n_cols - 1, col + 1)
        self.sim.robot_position = (row, col) 

        self.s = self.sim.get_world_state()

        print(self.sim.robot_position )
        if self.sim.robot_position == self.goal:
            self.s = self.sim.get_world_state()
            reward = self.calculate_reward()
            done = True
            info = {'prob': 1.0}
        else:
            if self.current_step >= self.max_steps:
                done = True
            else:
                done = False

            # Reset the robot position if done
            if done:
                self.sim.robot_position = self.start

            self.s = self.sim.get_world_state()
            bumped = self.s[2]  # Extract the bumped status from the world state
            reward = self.get_rewards(bumped)  # Use get_rewards for the bumped status
            info = {'prob': 1.0}

        return self.s, reward, done, False, info

    def get_rewards(self, bumped):
        """
        Calculates the reward for the current state.

        Parameters:
            bumped (str): The object the robot bumped into ("wall", "equipment", "worker", "target", or "").

        Returns:
            float: The reward based on the current state and the object bumped into.
        """
        # Assign rewards based on the severity of the bump
        if bumped == '':
            # addin positive reward if not hitting the wall or corners
            return 1.0          
        elif bumped == 'wall':
            # adding small penalty for hitting a wall
            return -5.0         
        elif bumped == 'equipment':
            # adding bigger 
            return -10.0       
        
        elif bumped == 'worker':
            # largest penalty for harming a human
            return -100.0       
        elif bumped == 'target':
            # reward for target
            return 50.0     
        else:
            return 0.0 

    def is_truncated(self):
        """
        Checks if the episode has reached the maximum number of steps.

        Returns:
            bool: True if the maximum number of steps has been reached, False otherwise.
        """
        return self.current_step >= self.max_steps

    def is_terminal(self, bumped):
        """
        Checks if the current state is terminal.

        Parameters:
            bumped (str): The object the robot bumped into ("wall", "equipment", "worker", "target", or "").

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        # terminate episode if hitting a woerkr
        if bumped == 'worker':
            return True 
        return False

    def render(self, close=False):
        """
        Render the environment.
        """
        if close and self.renderer:
            if self.renderer:
                self.renderer.close()
            return

        if self.renderer:
            return self.renderer.render()

    def encode_observation(self, world_state):
        """
        Encodes the given world state using linear indices for compactness.

        Parameters:
            world_state (tuple): The state of the world as returned by the get_world_state method.

        Returns:
            np.ndarray: The encoded state as a 1D numpy array.
        """
        # unpack the world state tuple
        row, col, bumped, n_boxes, prev_target, curr_target, workers_tuplist = world_state
        # encode the robot's position as a linear index
        robot_position = np.array([row * self.n_cols + col])
        # encode the bumped status
        bump_mapping = {"": 0, "wall": 1, "equipment": 2, "worker": 3, "target": 4}
        bumped_encoded = np.array([bump_mapping[bumped]])
        # encode the current target as a linear index
        curr_target_encoded = np.array([curr_target[0] * self.n_cols + curr_target[1]])
        # combine all encodings into a single vector
        encoded_state = np.concatenate([
            robot_position,
            bumped_encoded,
            curr_target_encoded,
        ])

        return encoded_state
