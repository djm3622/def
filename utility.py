import os

def validate_and_create_save_path(save_path, experiment_name):
    if experiment_name is None:
        raise ValueError("experiment_name must be set before validating save path")

    # Check if the path already exists
    if os.path.exists(save_path):
        raise AssertionError(f"Save path '{save_path}' already exists. Please choose a different experiment name.")

    # Create the directory and any necessary parent directories
    try:
        os.makedirs(save_path, exist_ok=False)
    except Exception as e:
        raise RuntimeError(f"Failed to create save directory: {str(e)}")