"""
Convert corrdinates to guassian heat map for each image as a GT for traning
Data: Sep 8, 2018
Author: Yucheng Tang
"""
import os
import os.path as path
import json
import numpy as np
from PIL import Image
from collections import Iterable
import argparse
import shutil 
import six
import string

parser = argparse.ArgumentParser(
    description="Convert coordinates to landmark gaussian heatmap",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--coord_dir", default="assignment1_data/labels",
    help="a folder that stores all the corrdinates infomation"
)
parser.add_argument(
    "--output_dir", default="assignment1_data/lmk_hm",
    help="a folder that stores all the image files"
)
parser.add_argument(
    "--dim", default=[326, 490],
    help="image dimensions"
)
args = parser.parse_args()

sigma = 3
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
label_file = os.path.join(args.coord_dir, 'labels.txt')

with open(label_file) as f:
    content = f.readlines()
content = [x.strip() for x in content]

for item in content:
    image_name = item.split('.')[0]
    lmk_x = round(float(item.split(' ')[1]) * args.dim[1])
    lmk_y = round(float(item.split(' ')[2]) * args.dim[0])
    heat_maps = []
    coordinates = np.mgrid[:args.dim[0], :args.dim[1]]
    distance_map = (coordinates[0] - lmk_y) ** 2 + (coordinates[1] - lmk_x) ** 2
    heat_map = np.exp(-distance_map / (2 * sigma ** 2))
    heat_map = np.array(heat_map).astype(np.float32) * 255.0
    hm_img = os.path.join(args.output_dir, image_name + '_hm.jpg')
    heat_map_img = Image.fromarray(heat_map).convert('RGB')
    heat_map_img.save(hm_img)