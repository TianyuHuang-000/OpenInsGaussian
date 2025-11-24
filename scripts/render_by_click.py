#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import json
from utils.opengs_utlis import mask_feature_mean, get_SAM_mask_and_feat, load_code_book
import pytorch3d.ops

np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0])
colors_defined = torch.from_numpy(colors_defined)

def get_pixel_values(image_path, position, radius=10):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size
        
        left = max(position[0] - radius, 0)
        right = min(position[0] + radius + 1, width)
        top = max(position[1] - radius, 0)
        bottom = min(position[1] + radius + 1, height)

        pixels = []
        for x in range(left, right):
            for y in range(top, bottom):
                pixels.append(img.getpixel((x, y)))

        pixels_array = np.array(pixels)
        mean_pixel = pixels_array.mean(axis=0)
    
    return tuple(mean_pixel)

def compute_click_values(model_path, image_name, pix_xy, radius=5):
    def compute_level_click_val(iter, model_path, image_name, pix_xy, radius):
        img_path1 = f"{model_path}/train/ours_{iter}/renders_ins_feat1/{image_name}_1.png"      # TODO
        img_path2 = f"{model_path}/train/ours_{iter}/renders_ins_feat2/{image_name}_2.png"      # TODO
        val1 = get_pixel_values(img_path1, pix_xy, radius)
        val2 = get_pixel_values(img_path2, pix_xy, radius)
        click_val = (torch.tensor(list(val1) + list(val2)) / 255) * 2 - 1
        return click_val
    
    level1_click_val = compute_level_click_val(50000, model_path, image_name, pix_xy, radius)   # TODO
    level2_click_val = compute_level_click_val(70000, model_path, image_name, pix_xy, radius)   # TODO
    
    return level1_click_val, level2_click_val

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat")
    gt_sam_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_sam_mask")
    pseudo_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pseudo_ins_feat")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_ins_feat_path, exist_ok=True)
    makedirs(gt_sam_mask_path, exist_ok=True)
    makedirs(pseudo_ins_feat_path, exist_ok=True)

    # load codebook
    root_code_book, root_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
        f'iteration_{iteration}', "root_code_book"))
    leaf_code_book, leaf_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
        f'iteration_{iteration}', "leaf_code_book"))
    root_cluster_indices = torch.from_numpy(root_cluster_indices).cuda()
    leaf_cluster_indices = torch.from_numpy(leaf_cluster_indices).cuda()
    # counts = torch.bincount(torch.from_numpy(cluster_indices), minlength=64)

    # load the saved codebook(leaf id) and instance-level language feature
    # 'leaf_feat', 'leaf_acore', 'occu_count', 'leaf_ind'       leaf_figurines_cluster_lang
    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    saved_data = np.load(mapping_file)
    leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()    # [num_leaf=640, 512] Language feature of each instance
    leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()       # [num_leaf=640] Score of each instance
    leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()  # [num_leaf=640] Number of occurrences of each instance
    leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()           # [num_pts] Instance ID corresponding to each point
    leaf_lang_feat[leaf_occu_count < 5] *= 0.0      # ignore
    leaf_cluster_indices = leaf_ind
    
    image_name = "frame_00002"      # TODO
    # # object_name = "apple"
    # pix_xy = (450, 217) # bag of cookies
    # pix_xy = (344, 350) # apple
    # # teatime       image_name = "frame_00002"
    # object_names = ["bear nose", "stuffed bear", "sheep", "bag of cookies", \
    #                 "plate", "three cookies", "tea in a glass", "apple", \
    #                 "coffee mug", "coffee", "paper napkin"]
    # pix_xy_list = [ (740, 80), (800, 160), (80, 240), (450, 200),
    #                 (468, 288), (438, 273), (309, 308), (343, 361),
    #                 (578, 274), (571, 260), (565, 380)]
    # figurines   image_name = "frame_00002"
    # TODO
    object_names = ["rubber duck with buoy", "porcelain hand", "miffy", "toy elephant", "toy cat statue", \
                    "jake", "Play-Doh bucket", "rubber duck with hat", "rubics cube", "waldo", \
                    "twizzlers", "red toy chair", "green toy chair", "pink ice cream", "spatula", \
                    "pikachu", "green apple", "rabbit", "old camera", "pumpkin", \
                    "tesla door handle"]
    # TODO
    pix_xy_list = [ (103, 378), (552, 390), (896, 342), (720, 257), (254, 297),
                    (451, 197), (626, 256), (760, 166), (781, 243), (896, 136),
                    (927, 241), (688, 148), (538, 160), (565, 238), (575, 257),
                    (377, 156), (156, 244), (21, 237), (283, 152), (330, 200),
                    (514, 200)]
    # # ramen           image_name = "frame_00002"
    # object_names = ["clouth", "sake cup", "chopsticks", "spoon", "plate", \
    #                 "bowl", "egg", "nori", "glass of water", "napkin"]
    # pix_xy_list = [(345, 38), (276, 424), (361, 370), (419, 285), (688, 412),
    #                (489, 119), (694, 187), (810, 154), (939, 289), (428, 462)]
    # # waldo_kitchen     image_name = "frame_00001"
    # object_names = ["knife", "pour-over vessel", "glass pot1", "glass pot2", "toaster", \
    #                 "hot water pot", "metal can", "cabinet", "ottolenghi", "waldo"]
    # pix_xy_list = [(439, 76), (410, 297), (306, 127), (349, 182), (261, 256),
    #                (201, 262), (161, 267), (80, 34), (17, 141), (76, 169)]

    for o_i, object in enumerate(object_names):
        pix_xy = pix_xy_list[o_i]
        root_click_val, leaf_click_val = compute_click_values(model_path, image_name, pix_xy)
    
        # Compute the nearest clusters with respect to the two-level codebook
        distances_root = torch.norm(root_click_val - root_code_book["ins_feat"][:, :-3].cpu(), dim=1)
        distances_leaf = torch.norm(leaf_click_val - leaf_code_book["ins_feat"][:-1, :].cpu(), dim=1)
        distances_leaf[leaf_code_book["ins_feat"][:-1].sum(-1) == 0] = 999  # Assign a large value to dis for nodes that remain unassigned
        
        # Retrieve the candidate child nodes linked to each selected root node
        min_index_root = torch.argmin(distances_root).item()
        leaf_num = (leaf_code_book["ins_feat"].shape[0] - 1) / root_code_book["ins_feat"].shape[0]
        start_id = int(min_index_root*leaf_num)
        end_id = int((min_index_root + 1)*leaf_num)
        distances_leaf_sub = distances_leaf[start_id: end_id]   # [10]

        # # (1) Choose several child nodes that fulfill the requirements
        # click_leaf_indices = torch.nonzero(distances_leaf_sub < 0.9).squeeze() + start_id
        # if (click_leaf_indices.dim() == 0) and click_leaf_indices.numel() != 0:
        #     click_leaf_indices = click_leaf_indices.unsqueeze(0) 
        # elif click_leaf_indices.numel() == 0:
        #     click_leaf_indices = torch.argmin(distances_leaf_sub).unsqueeze(0)
        # (2) identify the root-level codebook and then pick the closest leaf node inside it (preferred)
        click_leaf_indices = torch.argmin(distances_leaf_sub).unsqueeze(0) + start_id
        # (3) directly select the child node with the minimum distance (less precise)
        # click_leaf_indices = torch.argmin(distances_leaf).unsqueeze(0)
        # # (4) you can also directly specify a particular child node if needed
        # click_leaf_indices = torch.tensor([60, 66])     # 64 picachu, 60, 66 toy elephant, 65 jake, 633 green apple, 639 duck
        
        # Get the mask linked to the child node
        pre_pts_mask = (leaf_cluster_indices.unsqueeze(1) == click_leaf_indices.cuda()).any(dim=1)

        # post process  modify-----
        post_process = True
        max_time = 5
        if post_process and max_time > 0:
            nearest_k_distance = pytorch3d.ops.knn_points(
                gaussians._xyz[pre_pts_mask].unsqueeze(0),
                gaussians._xyz[pre_pts_mask].unsqueeze(0),
                K=int(pre_pts_mask.sum()**0.5) * 2,
            ).dists
            mean_nearest_k_distance, std_nearest_k_distance = nearest_k_distance.mean(), nearest_k_distance.std()
            # print(std_nearest_k_distance, "std_nearest_k_distance")

            # mask = nearest_k_distance.mean(dim = -1) < mean_nearest_k_distance + std_nearest_k_distance
            mask = nearest_k_distance.mean(dim = -1) < mean_nearest_k_distance + 0.1 * std_nearest_k_distance
            # mask = nearest_k_distance.mean(dim = -1) < 2 * mean_nearest_k_distance 

            mask = mask.squeeze()
            if pre_pts_mask is not None:
                pre_pts_mask[pre_pts_mask != 0] = mask
            max_time -= 1

        # out_dir = "ca9c2998-e"
        # splits = ["train", "train", "train", "train", "test"]
        # frame_name_list = ["frame_00053", "frame_00066", "frame_00140", "frame_00154", "frame_00089"]
        # for f_i, frame_name in enumerate(frame_name_list):
        #     base_path = f"/mnt/disk1/codes/wuyanmin/code/OpenGaussian/output/{out_dir}/{splits[f_i]}/ours_70000/renders_cluster_silhouette"
        #     target_path = f"/mnt/disk1/codes/wuyanmin/code/OpenGaussian/output/{out_dir}/{splits[f_i]}/ours_70000/result/{frame_name}"
        #     makedirs(target_path, exist_ok=True)
        #     for _, text in enumerate(waldo_kitchen_texts):
        #         pos_feat = text_features[query_texts.index(text)].unsqueeze(0)
        #         similarity_pos = F.cosine_similarity(pos_feat, leaf_lang_feat.cpu())    # [640]
        #         top_values, top_indices = torch.topk(similarity_pos, 10)   # [num_mask]
        #         print("text: {} | cluster id: {}".format(text, top_indices[0]))
        #         ori_img_name = base_path + f"/{frame_name}_cluster_{top_indices[0].item()}.png"
        #         new_name = target_path + f"/{text}.png"
                
        #         if not os.path.exists(ori_img_name):
        #             top = 10
        #             for i in range(top):
        #                 ori_img_name = target_path + f"/{frame_name}_cluster_{top_indices[i].item()}.png"
        #                 if os.path.exists(ori_img_name):
        #                     break
        #         if not os.path.exists(ori_img_name):
        #             print(f"No file found at {ori_img_name}. Operation skipped.")
        #             continue
        #         import shutil
        #         shutil.copy2(ori_img_name, new_name)

        # render
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # render_pkg = render(view, gaussians, pipeline, background, iteration, rescale=False)
            
            # # figurines
            # if  view.image_name not in ["frame_00041", "frame_00105", "frame_00152", "frame_00195"]:
            #     continue
            # # teatime
            # if  view.image_name not in ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]:
            #     continue
            # # ramen
            # if  view.image_name not in ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"]:
            #     continue
            # # waldo_kitchen
            # if  view.image_name not in ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"]:
            #     continue

            # NOTE render
            render_pkg = render(view, gaussians, pipeline, background, iteration,
                                rescale=False,                #)  # wherther to re-scale the gaussian scale
                                # cluster_idx=leaf_cluster_indices,     # root id 
                                leaf_cluster_idx=leaf_cluster_indices,            # leaf id               
                                selected_leaf_id=click_leaf_indices.cuda(),       # selected leaf id      
                                render_feat_map=True, 
                                render_cluster=False,
                                better_vis=True,
                                pre_mask=pre_pts_mask,
                                seg_rgb=True)
            rendering = render_pkg["render"]
            rendered_cluster_imgs = render_pkg["leaf_clusters_imgs"]
            occured_leaf_id = render_pkg["occured_leaf_id"]
            rendered_leaf_cluster_silhouettes = render_pkg["leaf_cluster_silhouettes"]

            # save Rendered RGB
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

            render_cluster_path = os.path.join(model_path, name, "ours_{}".format(iteration), "click_cluster")
            render_cluster_silhouette_path = os.path.join(model_path, name, "ours_{}".format(iteration), "click_cluster_mask")
            makedirs(render_cluster_path, exist_ok=True)
            makedirs(render_cluster_silhouette_path, exist_ok=True)
            for i, img in enumerate(rendered_cluster_imgs):
                torchvision.utils.save_image(img[:3,:,:], os.path.join(render_cluster_path, \
                    view.image_name + f"_{object}_cluster_{occured_leaf_id[i]}.png"))
                # save mask
                cluster_silhouette = rendered_leaf_cluster_silhouettes[i] > 0.8
                torchvision.utils.save_image(cluster_silhouette.to(torch.float32), os.path.join(render_cluster_silhouette_path, \
                    view.image_name + f"_{object}_cluster_{occured_leaf_id[i]}.png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)