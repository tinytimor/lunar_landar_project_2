import pygame
import os
import numpy as np
import scipy.stats as stats

class TrafficSim:
    """
    A class to simulate traffic at an intersection controlled by traffic lights.
    """

    def __init__(self, max_cars_dir, lambda_ns, lambda_ew, cars_leaving, ns, ew, light):
        """
        Initializes the TrafficSim object with the given parameters.

        Args:
            max_cars_dir (int): Maximum number of cars that can wait in a direction.
            lambda_ns (float): Rate parameter for Poisson distribution for cars approaching from the north-south direction.
            lambda_ew (float): Rate parameter for Poisson distribution for cars approaching from the east-west direction.
            cars_leaving (int): Number of cars that can leave the intersection when the light is green.
            ns (int): Initial number of cars waiting in the north-south direction.
            ew (int): Initial number of cars waiting in the east-west direction.
            light (int): Initial state of the north-south traffic light (1 for green, 0 for red).
        """

        # set the main parameters
        self.max_cars_dir = max_cars_dir
        self.lambda_ns = lambda_ns
        self.lambda_ew = lambda_ew
        self.cars_leaving = cars_leaving

        # set number of cars waiting and light at intersection
        self.cars_waiting_ns = ns
        self.cars_waiting_ew = ew
        self.light_ns = light

        # set probability of approaching cars
        self.prob_appr_cars = 1.0

    def get_approaching_cars(self):
        """
        Simulates the number of cars approaching the intersection from the north-south and east-west directions.

        Returns:
            tuple: Number of cars approaching from the north-south and east-west directions.
        """
        cars_appr_ns = np.random.poisson(self.lambda_ns)
        cars_appr_ew = np.random.poisson(self.lambda_ew)
        return cars_appr_ns, cars_appr_ew

    def get_updated_wait_cars(self, cars_wait_ns, cars_wait_ew, light, cars_appr_ns, cars_appr_ew):
        """
        Calculates the updated number of cars waiting in each direction after some cars have left the intersection.

        Args:
            cars_wait_ns (int): Current number of cars waiting in the north-south direction.
            cars_wait_ew (int): Current number of cars waiting in the east-west direction.
            light (int): Current state of the north-south traffic light (1 for green, 0 for red).
            cars_appr_ns (int): Number of cars approaching from the north-south direction.
            cars_appr_ew (int): Number of cars approaching from the east-west direction.

        Returns:
            tuple: Updated number of cars waiting in the north-south direction,
                   updated number of cars waiting in the east-west direction,
                   and the probability of the current combination of approaching cars.
        """
        # calculate the updated the number of cars in the N/S and E/W directions
        updated_cars_wait_ns = min(max(cars_wait_ns + cars_appr_ns - light*self.cars_leaving, 0), self.max_cars_dir)
        updated_cars_wait_ew = min(max(cars_wait_ew + cars_appr_ew - (1-light)*self.cars_leaving, 0), self.max_cars_dir)
        # calculate the probability of the combo of approaching cars
        prob_appr_ns = stats.poisson.pmf(cars_appr_ns, self.lambda_ns)
        prob_appr_ew = stats.poisson.pmf(cars_appr_ew, self.lambda_ew)
        return updated_cars_wait_ns, updated_cars_wait_ew, prob_appr_ns * prob_appr_ew

    def advance(self, action):
        """
        Advances the state of the world by one timestep based on the given action.

        Args:
            action (int): The action taken by the agent (1 for changing the light, 0 for keeping it the same).
        """
        # get number of cars approaching in the N/S and E/W directions
        cars_appr_ns, cars_appr_ew = self.get_approaching_cars()
        # update the traffic light state
        self.light_ns = abs(self.light_ns - action)
        # get the updated number of cars waiting
        updated_cars_wait_ns, updated_cars_wait_ew, prob_appr_cars  = self.get_updated_wait_cars(self.cars_waiting_ns, self.cars_waiting_ew, self.light_ns, cars_appr_ns, cars_appr_ew)
        # update the number of cars in the N/S direction
        self.cars_waiting_ns = updated_cars_wait_ns
        # update the number of cars in the E/W direction
        self.cars_waiting_ew = updated_cars_wait_ew
        # update probability of approaching cars
        self.prob_appr_cars = prob_appr_cars

    def get_world_state(self):
        """
        Retrieves the current state of the world.

        Returns:
            tuple: Current number of cars waiting in the north-south direction,
                   current number of cars waiting in the east-west direction,
                   current state of the north-south traffic light,
                   and the probability of the current combination of approaching cars.
        """
        return self.cars_waiting_ns, self.cars_waiting_ew, self.light_ns, self.prob_appr_cars

    def reset(self, ns, ew, light):
        """
        Resets the world to the desired state with the given parameters.

        Args:
            ns (int): Desired number of cars waiting in the north-south direction.
            ew (int): Desired number of cars waiting in the east-west direction.
            light (int): Desired state of the north-south traffic light (1 for green, 0 for red).
        """
        self.cars_waiting_ns = ns
        self.cars_waiting_ew = ew
        self.light_ns = light
        self.prob_appr_cars = 1.0


class TrafficRenderer:
    """
    A class to render the traffic simulation environment using Pygame.

    Attributes:
        sim (TrafficSim): The traffic simulation object to render.
        mode (str): The rendering mode, either 'human' or 'rgb_array'.
        screen_width (int): Width of the rendering window.
        screen_height (int): Height of the rendering window.
        window (pygame.Surface): The Pygame window surface.
        screen (pygame.Surface): The Pygame screen surface for rendering.
        background_image (pygame.Surface): Background image for the intersection.
        car_im_ns (pygame.Surface): Image for cars moving north to south.
        car_im_sn (pygame.Surface): Image for cars moving south to north.
        car_im_ew (pygame.Surface): Image for cars moving east to west.
        car_im_we (pygame.Surface): Image for cars moving west to east.
        traffic_light_green (pygame.Surface): Image for green traffic light.
        traffic_light_red (pygame.Surface): Image for red traffic light.
        end_image (pygame.Surface): Image displayed at the end of the simulation.
    """

    def __init__(self, sim, mode):
        """
        Initializes the TrafficRenderer object with the given simulation and rendering mode.

        Args:
            sim (TrafficSim): The traffic simulation object to render.
            mode (str): The rendering mode, either 'human' or 'rgb_array'.
        """
        # store simulator object
        self.sim = sim
        # store render mode
        self.mode = mode

        # initialize pygame
        pygame.init()
        self.screen_width = 1082
        self.screen_height = 1084

        # set the window to be resizable
        self.window = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        self.screen = pygame.Surface((self.screen_width, self.screen_height))

        # load background, light and car images
        self.background_image = pygame.image.load(os.path.join("images", "intersection.png"))
        self.car_im_ns = pygame.image.load(os.path.join("images", "car_ns.png"))
        self.car_im_sn = pygame.image.load(os.path.join("images", "car_sn.png"))
        self.car_im_ew = pygame.image.load(os.path.join("images", "car_ew.png"))
        self.car_im_we = pygame.image.load(os.path.join("images", "car_we.png"))
        self.traffic_light_green = pygame.image.load(os.path.join("images", "light_green.png"))
        self.traffic_light_red = pygame.image.load(os.path.join("images", "light_red.png"))
        self.end_image = pygame.image.load(os.path.join("images", "simulation_over.png"))

    def render(self, ns, ew, light):
        """
        Renders the current state of the traffic simulation environment.

        Args:
            ns (int): Number of cars waiting in the north-south direction.
            ew (int): Number of cars waiting in the east-west direction.
            light (int): State of the north-south traffic light (1 for green, 0 for red).

        Returns:
            None or np.ndarray: If mode is 'human', displays the rendering
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.VIDEORESIZE:
                # handle window resizing
                self.screen_width, self.screen_height = event.size
                self.window = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        # draw background
        self.screen.blit(self.background_image, (0, 0))
        # draw North-South cars
        y_offset = 415
        for i in range(ns//2):
            self.screen.blit(self.car_im_ns, (515, y_offset - i * (self.car_im_ns.get_height() + 10)))
        # draw South-North cars
        y_offset = 625
        for i in range(ns - ns//2):
            self.screen.blit(self.car_im_sn, (545, y_offset + i * (self.car_im_sn.get_height() + 10)))
        # draw East-West cars
        x_offset = 625
        for i in range(ew//2):
            self.screen.blit(self.car_im_ew, (x_offset + i * (self.car_im_ew.get_width() + 10), 515))
        # draw West-East cars
        x_offset = 415
        for i in range(ew - ew//2):
            self.screen.blit(self.car_im_we, (x_offset - i * (self.car_im_we.get_width() + 10), 545))
        # draw traffic lights
        position_light_ns = (415, 305)
        position_light_ew = (685, 600)
        if light == 1:  # North-South green, East-West red
            self.screen.blit(self.traffic_light_green, position_light_ns)  # North-South traffic light
            self.screen.blit(self.traffic_light_red, position_light_ew)    # East-West traffic light
        else:  # North-South red, East-West green
            self.screen.blit(self.traffic_light_red, position_light_ns)    # North-South traffic light
            self.screen.blit(self.traffic_light_green, position_light_ew)  # East-West traffic light
        # select different modes
        if self.mode == 'human':
            self.window.blit(self.screen, (0, 0))
            pygame.display.update()
        elif self.mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.quit()
