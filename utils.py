import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import yaml
from types import SimpleNamespace


def deprocess(img):
    return img * 127.5 + 127.5

def plot_batch(tensor, plot_shape, filename, img_size=32):
    tensor_np = tensor.permute(0, 2, 3, 1).numpy()
    tensor_np = np.clip(tensor_np, 0, 255).astype(np.uint8)
    rows = plot_shape[0]
    columns = plot_shape[1]
    # Proportional scale by img size
    if isinstance(img_size, (tuple, list)):
        scale = (max(img_size) / 90)
    else:
        scale = (img_size / 90)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * scale, rows * scale))
    
    # Iterate through each cell in the grid and plot images
    for i in range(rows):
        for j in range(columns):
            # Calculate the index of the image
            img_idx = i * columns + j
            
            # If the image index exceeds the number of images, stop plotting
            if img_idx >= tensor_np.shape[0]:
                break
            
            # Display the image in the respective subplot
            ax = axes[i, j]
            ax.imshow(tensor_np[img_idx])
            ax.axis('off')
    
    # Adjust the layout to minimize whitespace
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(filename)
    plt.close()

class Config(object):
    def __init__(self, input_dict, save_dir):
        file_path = os.path.join(save_dir, "config.json")
        # Check if the configuration file exists
        if os.path.exists(file_path):
            self.load_config(file_path)
        else:
            for key, value in input_dict.items():
                setattr(self, key, value)
            self.save_config(file_path, save_dir)
        self.print_variables()

    def save_config(self, file_path, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert input_dict to JSON and save to file
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)
        print(f'New config {file_path} saved')

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            setattr(self, key, value)
        print(f'Config {file_path} loaded')

    def print_variables(self):
        # Print all variables (attributes) of the Config object
        for key, value in vars(self).items():
            print(f"{key}: {value}")


def load_config_from_yaml(yaml_path, save_dir):
    """
    Load configuration from YAML file and return a Config object.
    Uses SimpleNamespace to convert YAML dict to object with attribute access.
    """
    file_path = os.path.join(save_dir, "config.json")
    
    # Check if the configuration file exists (resume from checkpoint)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            config_data = json.load(f)
        config = SimpleNamespace(**config_data)
        print(f'Config {file_path} loaded')
    else:
        # Load from YAML
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)
        
        # Save config as JSON
        os.makedirs(save_dir, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(vars(config), f, indent=4)
        print(f'New config {file_path} saved from {yaml_path}')
    
    # Print all variables
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    
    return config


def get_next_model_dir(base_dir='output'):
    """
    Get the next available model directory name in the format output/model_i
    where i is the next number after the highest existing model number.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_models = []
    
    for name in os.listdir(base_dir):
        if name.startswith('model_'):
            try:
                num = int(name.split('_')[1])
                existing_models.append(num)
            except (ValueError, IndexError):
                continue
    
    next_num = max(existing_models) + 1 if existing_models else 1
    return os.path.join(base_dir, f'model_{next_num}')

