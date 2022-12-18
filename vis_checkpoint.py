"""Visualize model checkpoints and inferences"""
import os
import sys
from models import unet256, unet512
from main import load_config
import Experiment

if __name__ == "__main__":
    CONFIG_DIR = "/local_directory/lobe_seg/configs"
    config_id = sys.argv[1]
    model_name = sys.argv[2]
    config = load_config(f"{config_id}.YAML", CONFIG_DIR)

    exp = Experiment.Experiment(config, config_id, unet256)
    exp.vis_model(model_name, nvals=50)