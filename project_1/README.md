# CS7642

Contains the two folders for the first CS7642 project containg `gymRobot-templates` and `gymTraffic-templates` direectories. Each environment simulates a different scenario and includes the necessary code to run and train agents within these environments.

## gymRobot-templates

### Overview

The `gymRobot-templates` directory contains a custom Gym environment simulating a warehouse where a robot must navigate to deliver boxes while avoiding obstacles and workers.

### Files

- `robot_environment.py`: Defines the `RobotEnv` class, which sets up the environment and handles the simulation logic.
- `robot_simulator.py`: Contains the `RobotSim` class, which simulates the warehouse and the robot's interactions within it.
- `robot_execution.py`: Script to run the robot environment and train an agent.
- `rl_agents.py`: Contains the reinforcement learning agents (QLearningAgent and SARSAgent) used to train the robot.
- `rl_planners.py`: Contains planning algorithms (ValueIterationPlanner and PolicyIterationPlanner) for the robot environment.

### How to Run

1. Change to the `gymRobot-templates` directory:
    ```bash
    cd gymRobot-templates
    ```

2. Ensure you have the required dependencies installed:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the robot environment:
    ```bash
    python robot_environment.py
    ```

4. To execute the training script:
    ```bash
    python robot_execution.py
    ```

### Note

The code may contain bugs and could break during execution. If you encounter any issues, please check the code and debug as necessary.

## gymTraffic-templates

### Overview

The `gymTraffic-templates` directory contains a custom Gym environment simulating a traffic intersection controlled by traffic lights. The goal is to manage the traffic flow efficiently.

### Files

- `traffic_environment.py`: Defines the `TrafficEnv` class, which sets up the environment and handles the simulation logic.
- `traffic_simulator.py`: Contains the `TrafficSim` class, which simulates the traffic intersection and the traffic light's interactions.
- `traffic_execution.py`: Script to run the traffic environment and train an agent.
- `rl_agents.py`: Contains the reinforcement learning agents (QLearningAgent and SARSAgent) used to train the traffic light controller.
- `rl_planners.py`: Contains planning algorithms (ValueIterationPlanner and PolicyIterationPlanner) for the traffic environment.

### How to Run

1. Change to the `gymTraffic-templates` directory:
    ```bash
    cd gymTraffic-templates
    ```

2. Ensure you have the required dependencies installed:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the traffic environment:
    ```bash
    python traffic_environment.py
    ```

4. To execute the training script:
    ```bash
    python traffic_execution.py
    ```

### Note

The code may contain bugs and could break during execution. If you encounter any issues, please check the code and debug as necessary.

## Dependencies

Make sure to install the required dependencies for both environments. You can do this by running:
