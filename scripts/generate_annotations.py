# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time  # Add this import at the top of the file
import json  # Add this import at the top of the file
from typing import Optional, Tuple

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    # Speed bump
    # img_folder = '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/jiuting_campus'
    # file_names = ['20250331_111636.639_10.png', '20250331_111635.380_10.png', '20250331_111634.669_10.png',
    #               '20250331_111634.138_10.png'] 

    #big hole
    # img_folder = '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/lidar'
    # file_names = ['00000051.png', '00000049.png', '00000060.png',
    #               '00000061.png'] 
    
    img_folder = '/media/levin/DATA/nerf/new_es8/stereo/250610/'
    file_names = ['00000005.png', '00000006.png',
                  '00000011.png', '00000012.png',
                  '00000018.png', '00000023.png', '00000024.png']
    # file_names = []
    if not file_names:
        rgb_folder = f"{img_folder}/colored_l/"
        file_names = [os.path.basename(f) for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg'))]

    file_names.sort()
    # Big hole
    # img_folder = '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/lidar'
    # file_name = '00000062.png'
    args.intrinsic_file = "/media/levin/DATA/nerf/new_es8/stereo_20250331/K_Zed.txt"

    args.z_far = 100
    args.denoise_cloud = False
    crop_y = 1000

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()

    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
        baseline = float(lines[1])

    annotations = {"files": []}  # Initialize the annotations dictionary

    # Loop through each file in the list
    for file_name in file_names:
        args.left_file = f"{img_folder}/colored_l/{file_name}"
        args.right_file = f"{img_folder}/colored_r/{file_name}"
        code_dir = os.path.dirname(os.path.realpath(__file__))
        img0 = imageio.v2.imread(args.left_file, pilmode="RGB")
        img1 = imageio.v2.imread(args.right_file, pilmode="RGB")
        scale = args.scale
        assert scale <= 1, "scale must be <=1"
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        H, W = img0.shape[:2]
        img0_ori = img0.copy()
        # logging.info(f"img0: {img0.shape}")

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        start_time = time.time()  # Start timing
        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        end_time = time.time()  # End timing

        # print(f"Running duration: {end_time - start_time:.2f} seconds")  # Print the duration

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0
        disp[invalid] = np.inf

        K[:2] *= scale
        depth = K[0, 0] * baseline / disp
        
        depth[depth< 0] = 0
        depth[depth > 100] = 0
        # import matplotlib.pyplot as plt

        # # Display the RGB image and depth image side by side
        # plt.figure(figsize=(12, 6))

        # plt.subplot(1, 2, 1)
        # plt.title("RGB Image")
        # plt.imshow(img0_ori)
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.title("Depth Image")
        # plt.imshow(depth)
        # plt.colorbar(label="Depth (meters)")
        # plt.axis("off")

        # plt.tight_layout()
        # # plt.savefig(f'{args.out_dir}/rgb_and_depth.png')
        # plt.show()

        # Convert intrinsic matrix K to [fx, fy, cx, cy] format
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        intrinsic_params = [fx, fy, cx, cy]  # Ensure all values are native Python floats

        depth_scale = 1.0

        # Define the base annotation path
        annotation_base_path = f'{img_folder}/annotation'

        # Save RGB image
        output_rgb_dir = f'{annotation_base_path}/rgb'
        os.makedirs(output_rgb_dir, exist_ok=True)  # Ensure the directory exists
        rgb_file_path = f'rgb/{file_name}'  # Relative path
        imageio.imwrite(f'{annotation_base_path}/{rgb_file_path}', img0_ori[:crop_y, :, :])
        logging.info(f"rgb saved to {annotation_base_path}/{rgb_file_path}")

        # Save depth file
        output_depth_dir = f'{annotation_base_path}/depth'
        os.makedirs(output_depth_dir, exist_ok=True)  # Ensure the directory exists
        npy_file_name = file_name.replace('png', 'npy')
        depth_file_path = f'depth/{npy_file_name}'  # Relative path
        np.save(f'{annotation_base_path}/{depth_file_path}', depth[:crop_y, :])
        # process_and_visualize_point_cloud(depth, K, img0_ori, args.out_dir, args.z_far, args.denoise_cloud, args.denoise_nb_points, args.denoise_radius)

        # Add annotation information for the current file
        annotations["files"].append({
            "cam_in": intrinsic_params,  # Already converted to a valid list of floats
            "rgb": rgb_file_path,  # Relative path
            "depth": depth_file_path,  # Relative path
            "depth_scale": depth_scale
        })

    # Save annotations to a JSON file
    json_file_path = os.path.join(annotation_base_path, "zed_annotation.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)
    logging.info(f"Annotations saved to {json_file_path}")

