# Project Title

This project is designed to train and evaluate a reinforcement learning model using the Overcooked environment. The project is implemented in Python 3.8 and utilizes several Python libraries for training, evaluation, and plotting.

## Installation

To get started, ensure you have Python 3.8 installed on your system. Then, follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.gatech.edu/gt-omscs-rldm/7642RLDMFall2024slehman7.git
   cd project_3
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## File Descriptions

### `training.py`

- **Purpose**: This script is responsible for training the reinforcement learning model using the Overcooked environment.
- **Key Components**:
  - Initializes the environment and model architectures.
  - Implements a curriculum learning strategy to train the model across different layouts.
  - Saves checkpoints and training data periodically.
- **How to Run**:
  ```bash
  python training.py
  ```
- **Outputs**:
  - Checkpoints are saved in the `checkpoints` directory.
  - Training data is saved in the `training_data` directory and as `training_data.csv`.

### `evaluation.py`

- **Purpose**: This script evaluates the trained model using saved checkpoints.
- **Key Components**:
  - Loads checkpoints and evaluates the model across different layouts.
  - Generates plots and saves evaluation results.
- **How to Run**:
  ```bash
  python evaluation.py
  ```
- **Outputs**:
  - Evaluation results are saved in the `evaluation_results` directory as CSV and pickle files.
  - Performance plots are saved in the `plots` directory.

### `metrics.py`

- **Purpose**: Contains utility functions to extract and compute various metrics during training and evaluation.
- **Key Components**:
  - Functions to calculate metrics related to onion, dish, and soup interactions.
  - Computes reward metrics and collaboration metrics.

### `plotting.py`

- **Purpose**: Provides functions to generate plots for visualizing training and evaluation metrics.
- **Key Components**:
  - Functions to plot metrics such as soups made, rewards, and other performance indicators.

## Running the Project

1. **Training**: Execute the `training.py` script to start training the model. The script will automatically save checkpoints and training data.

2. **Evaluation**: After training, run the `evaluation.py` script to evaluate the model using the saved checkpoints. This will generate evaluation results and plots.

## Using Trained Models

To use the trained models, follow these steps:

1. **Locate Checkpoints**: Ensure that your trained model checkpoints are saved in the `checkpoints` directory. These files should have a `.pth` extension.

2. **Load and Evaluate**: Use the `evaluation.py` script to load and evaluate the models. The script will automatically load the checkpoints and perform evaluations across different layouts.

3. **Command**: Run the following command to start the evaluation:
   ```bash
   python evaluation.py
   ```

4. **Results**: The evaluation results will be saved in the `evaluation_results` directory as CSV and pickle files. Performance plots will be saved in the `plots` directory.

## Outputs and Data Storage

- **Checkpoints**: Saved in the `checkpoints` directory. These files contain the model's state at various points during training.
- **Training Data**: Stored in the `training_data` directory and as `training_data.csv`.
- **Evaluation Results**: Saved in the `evaluation_results` directory as CSV and pickle files.
- **Plots**: Generated plots are saved in the `plots` directory, providing visual insights into the model's performance.

## Additional Information

- Ensure that the `overcooked_ai` package is correctly installed and accessible, as it is a critical dependency for the environment setup.
- The project uses PyTorch for model implementation and training, so ensure CUDA is available if you wish to leverage GPU acceleration.

For any issues or questions, please refer to the project's documentation or contact the maintainers.