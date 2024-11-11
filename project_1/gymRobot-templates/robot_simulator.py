import pygame
import numpy as np
import random

class RobotSim:

    def __init__(self, n_rows, n_cols, warehouse_workers, warehouse_equipment, target, start):
        """
        Initializes the RobotSim environment.

        Parameters:
            n_rows (int): The number of rows in the warehouse grid.
            n_cols (int): The number of columns in the warehouse grid.
            warehouse_workers (list): List of initial positions of warehouse workers as (row, col) tuples.
            warehouse_equipment (list): List of positions of warehouse equipment as (row, col) tuples.
            target (tuple): The target position where the robot needs to deliver boxes.
            row (int): Initial row position of the robot.
            col (int): Initial column position of the robot.
        """
        # set the warehouse layout
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.warehouse_workers = np.array(warehouse_workers)
        self.og_warehouse_workers = np.array(warehouse_workers)
        self.warehouse_equipment = np.array(warehouse_equipment)

        # set the number of boxes delivered
        self.n_boxes = 0

        # get all locations on the warehouse
        self.all_locations = np.indices((self.n_rows, self.n_cols)).transpose(1, 2, 0).reshape(-1, 2)
        self.row, self.col = start
        self.update_unreachable_and_reachable_locations()

        # define the previous and current target for delivery
        self.prev_target = None
        self.curr_target = target

        # define whether the robot has bumped into anything
        self.bumped = ""

        # define the effect of different actions the agent can take
        self.action_effects = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }

        # store the simulation iteration
        self.iteration = 0

    def update_unreachable_and_reachable_locations(self):
        """
        Updates the unreachable and reachable locations in the warehouse based on the current state.
        """
        self.unreachable_locations = self.get_unreachable_locations()
        self.reachable_locations = self.get_reachable_locations()

    def get_unreachable_locations(self):
        """
        Identifies locations that are unreachable by the robot.

        Returns:
            np.ndarray: A 2D array of positions that are unreachable, including the robot's current position, workers, and equipment.
        """
        # includes the robot's current position as unreachable, along with workers and equipment
        return np.vstack((np.array([self.row, self.col]), self.warehouse_workers, self.warehouse_equipment))

    def get_reachable_locations(self):
        """
        Identifies locations that are reachable by the robot.

        Returns:
            np.ndarray: A 2D array of positions that are reachable by the robot.
        """
        return np.array([loc for loc in self.all_locations if not np.any(np.all(loc == self.unreachable_locations, axis=1))])

    def move_workers(self):
        """
        Moves the warehouse workers randomly within one step of their original positions, if the movement is valid.
        Updates the unreachable and reachable locations after moving the workers.
        """
        # for each worker on the array, try to move them randomly by no more than a step away from their orignal_position
        for worker_index in range(len(self.warehouse_workers)):
            orignal_position = self.og_warehouse_workers[worker_index]
            effect_index = random.choice([0,1,2,3])
            effect = self.action_effects[effect_index]
            next_position = self.warehouse_workers[worker_index] + np.array(effect)
            # check if the new position is within bounds
            if (0 <= next_position[0] < self.n_rows) and (0 <= next_position[1] < self.n_cols):
                # check if the new position is in reachable locations
                if np.any(np.all(next_position == self.reachable_locations, axis=1)) and np.sum(np.abs(next_position-orignal_position)) < 2:
                    self.warehouse_workers[worker_index] = next_position
        # update unreachable and reachable locations after workers move
        self.update_unreachable_and_reachable_locations()

    def move_robot(self, action):
        """
        Moves the robot based on the given action and updates the state.

        Parameters:
            action (int): The action to take, where 0 = up, 1 = right, 2 = down, and 3 = left.

        Returns:
            tuple: The new position of the robot as (row, col).
            str: The object the robot bumped into, if any ("wall", "equipment", "worker", or "").
        """
        # get and apply the effect of the action
        effect = self.action_effects[action]
        next_state = np.array([self.row, self.col]) + np.array(effect)
        # handle boundary conditions
        if not (0 <= next_state[0] < self.n_rows) or not (0 <= next_state[1] < self.n_cols):
            bumped = "wall"
            next_state -= np.array(effect)
        # handle the possibility of bumping onto the storage bays
        elif np.any(np.all(self.warehouse_equipment == next_state, axis=1)):
            bumped = "equipment"
            next_state -= np.array(effect)
        # handle the possibility of bumping onto a person
        elif np.any(np.all(self.warehouse_workers == next_state, axis=1)):
            bumped = "worker"
        else:
            bumped = ""
        # updates the state
        self.row = next_state[0]
        self.col = next_state[1]
        self.bumped = bumped
        # update unreachable and reachable locations after the robot moves
        self.update_unreachable_and_reachable_locations()
        return tuple(next_state), bumped

    def get_new_target(self):
        """
        Randomly selects a new target from the reachable locations.

        Returns:
            tuple: The new target position as (row, col).
        """
        random_index = np.random.choice(len(self.reachable_locations))
        return self.reachable_locations[random_index]

    def advance(self, action):
        """
        Advances the simulation by moving the robot and workers based on the given action.

        Parameters:
            action (int): The action to take, where 0 = up, 1 = right, 2 = down, and 3 = left.
        """
        # update current simulation step
        self.iteration += 1
        # move robot based on the action from the agent
        next_state, bumped = self.move_robot(action)
        # check if the robot delivered a box
        new_delivery = self.is_box_delivered()
        # update the new target if a box was delivered
        if new_delivery:
            self.prev_target = tuple(self.curr_target)
            self.curr_target = tuple(self.get_new_target())
        # move workers randomly
        self.move_workers()

    def is_box_delivered(self):
        """
        Checks if the robot has delivered a box to the target.

        Returns:
            bool: True if the robot is at the target and a box was delivered, False otherwise.
        """
        if (self.row, self.col) == self.curr_target:
            self.bumped = "target"
            self.n_boxes += 1
            return True
        else:
            return False

    def workers_tuplist(self):
        """
        Converts the list of warehouse workers' positions from a numpy array to a list of tuples.

        Returns:
            list of tuple: A list where each tuple represents the (row, col) position of a warehouse worker.
        """
        return [tuple(worker) for worker in self.warehouse_workers.tolist()]

    def get_world_state(self):
        """
        Retrieves the current state of the world.

        Returns:
        tuple: The robot's current row and column, the last object it bumped into, the number of boxes delivered,
        the previous and current target positions, and a list of tuples representing the positions of the warehouse workers.
        """
        return self.row, self.col, self.bumped, self.n_boxes, self.prev_target, self.curr_target, self.workers_tuplist()

    def reset(self, n_rows, n_cols, warehouse_workers, warehouse_equipment, target, start):
        """
        Resets the simulation to the initial state.

        Parameters:
            n_rows (int): The number of rows in the warehouse grid.
            n_cols (int): The number of columns in the warehouse grid.
            warehouse_workers (list): List of initial positions of warehouse workers as (row, col) tuples.
            warehouse_equipment (list): List of positions of warehouse equipment as (row, col) tuples.
            target (tuple): The target position where the robot needs to deliver boxes.
            row (int): Initial row position of the robot.
            col (int): Initial column position of the robot.
        """
        self.iteration = 0
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.warehouse_workers = np.array(warehouse_workers)
        self.warehouse_equipment = np.array(warehouse_equipment)
        self.n_boxes = 0
        self.row = start[0]
        self.col = start[1]
        self.prev_target = None
        self.curr_target = target

import pygame

class RobotRenderer:
    def __init__(self, sim, screen_width=800, screen_height=800, mode='human'):
        """
        Initializes the rendering environment.

        Parameters:
            sim (RobotSim): The robot simulation instance to be rendered.
            screen_width (int, optional): The width of the window screen. Default is 800.
            screen_height (int, optional): The height of the window screen. Default is 800.
            mode (str, optional): The rendering mode, either 'human' for interactive display or 'rgb_array' for array output. Default is 'human'.
        """
        # store simulator object and render mode
        self.sim = sim
        self.mode = mode

        # initialize pygame
        pygame.init()

        # get the number of rows and columns from the simulator
        self.n_rows = self.sim.n_rows
        self.n_cols = self.sim.n_cols

        # screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height

        # calculate cell size dynamically
        self.cell_size = min(self.screen_width // self.n_cols, self.screen_height // self.n_rows)

        # create the pygame window and screen
        self.window = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        self.screen = pygame.Surface((self.screen_width, self.screen_height))

        # create images
        self.background_image = pygame.Surface((self.screen_width, self.screen_height))
        self.background_image.fill((155, 155, 155))  # grey background

        self.robot_image = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.circle(self.robot_image, (0, 128, 0), (self.cell_size // 2, self.cell_size // 2), self.cell_size // 2)

        self.goal_image = pygame.Surface((self.cell_size, self.cell_size))
        self.goal_image.fill((128, 0, 128))  # purple square as the goal

        self.worker_image = pygame.Surface((self.cell_size, self.cell_size))
        self.worker_image.fill((255, 0, 0))  # red square for workers

        self.equipment_image = pygame.Surface((self.cell_size, self.cell_size))
        self.equipment_image.fill((75, 75, 75))  # grey square for equipment

    def draw_grid(self):
        """
        Draws the grid lines on the screen to represent the warehouse layout.
        """
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.screen_width, y))

    def render(self):
        """
        Renders the entire environment, including the robot, workers, equipment, and target.

        Returns:
            np.ndarray (optional): The RGB array of the screen if mode is 'rgb_array'.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resizing
                self.screen_width, self.screen_height = event.size
                self.window = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
                self.cell_size = min(self.screen_width // self.n_cols, self.screen_height // self.n_rows)

        # draw background and grid
        self.screen.blit(self.background_image, (0, 0))
        self.draw_grid()

        # draw goal
        goal_position = (self.sim.curr_target[1] * self.cell_size, self.sim.curr_target[0] * self.cell_size)
        self.screen.blit(self.goal_image, goal_position)

        # draw workers
        for ww_index in range(len(self.sim.warehouse_workers)):
            ww = self.sim.warehouse_workers[ww_index]
            ww_position = (ww[1] * self.cell_size, ww[0] * self.cell_size)
            self.screen.blit(self.worker_image, ww_position)

        # draw equipment
        for we_index in range(len(self.sim.warehouse_equipment)):
            we = self.sim.warehouse_equipment[we_index]
            we_position = (we[1] * self.cell_size, we[0] * self.cell_size)
            self.screen.blit(self.equipment_image, we_position)

        # draw robot
        robot_position = (self.sim.col * self.cell_size, self.sim.row * self.cell_size)
        self.screen.blit(self.robot_image, robot_position)

        if self.mode == 'human':
            self.window.blit(self.screen, (0, 0))
            pygame.display.update()
        elif self.mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        """
        Closes the Pygame window and quits the Pygame environment.
        """
        pygame.quit()
