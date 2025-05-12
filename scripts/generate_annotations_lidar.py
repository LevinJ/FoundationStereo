# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import json  # Add this import at the top of the file
import numpy as np

# Speed bump
img_folder = '/media/levin/DATA/nerf/new_es8/stereo/zed_lidar1/depth'
file_names = [] 

if not file_names:
    rgb_folder = f"{img_folder}/rgb"
    file_names = [f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg'))]


intrinsic_file = "/media/levin/DATA/nerf/new_es8/stereo_20250331/K_Zed.txt"


K = np.eye(3, dtype=np.float32)
with open(intrinsic_file, 'r') as f:
    lines = f.readlines()
    K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)

annotations = {"files": []}  # Initialize the annotations dictionary

# Loop through each file in the list
for file_name in file_names:

    # Convert intrinsic matrix K to [fx, fy, cx, cy] format
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    intrinsic_params = [fx, fy, cx, cy]  # Ensure all values are native Python floats

    depth_scale = 256.0
   
   
    rgb_file_path = f'rgb/{file_name}' 
    depth_file_path = f'depth/{file_name}' 

    # Add annotation information for the current file
    annotations["files"].append({
        "cam_in": intrinsic_params,  # Already converted to a valid list of floats
        "rgb": rgb_file_path,  # Relative path
        "depth": depth_file_path,  # Relative path
        "depth_scale": depth_scale
    })

# Save annotations to a JSON file
json_file_path = os.path.join(img_folder, "zed_annotation.json")
with open(json_file_path, 'w') as json_file:
    json.dump(annotations, json_file, indent=4)
print(f"Annotations saved to {json_file_path}")

