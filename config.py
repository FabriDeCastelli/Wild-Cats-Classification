import os

# Path to the project folder
PROJECT_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# Path to the hyperparameters folder
HYPERPARAMETERS_PATH = PROJECT_FOLDER_PATH + "/hyperparameters/{}.yaml"

RESULTS_DIRECTORY = PROJECT_FOLDER_PATH + "/results/{}"

TENSORBOARD_LOGS = PROJECT_FOLDER_PATH + "/logs/{}"

