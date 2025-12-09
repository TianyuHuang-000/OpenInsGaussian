import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

def load_image_as_binary(image_path, is_png=False, threshold=10):
    image = Image.open(image_path)
    if is_png:
        image = image.convert('L')
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(int)
    return binary_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

import os
import numpy as np

def evalute(gt_base, pred_base, scene_name, output_file="evaluation_results.txt"):
    scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
    }
    frame_names = scene_gt_frames[scene_name]

    ious = []
    with open(output_file, "w") as f:
        for frame in frame_names:
            f.write(f"frame: {frame}\n")
            gt_floder = os.path.join(gt_base, frame)
            file_names = [f for f in os.listdir(gt_floder) if f.endswith('.jpg')]
            for file_name in file_names:
                base_name = os.path.splitext(file_name)[0]
                gt_obj_path = os.path.join(gt_floder, file_name)
                pred_obj_path = os.path.join(pred_base, frame + "_" + base_name + '.png')
                if not os.path.exists(pred_obj_path):
                    f.write(f"Missing pred file for {file_name}, skipping...\n")
                    f.write(f"IoU for {file_name}: 0\n")
                    ious.append(0.0)
                    continue
                mask_gt = load_image_as_binary(gt_obj_path)
                mask_pred = load_image_as_binary(pred_obj_path, is_png=True)
                iou = calculate_iou(mask_gt, mask_pred)
                ious.append(iou)
                f.write(f"IoU for {file_name} and {base_name + '.png'}: {iou:.4f}\n")

        # Acc.
        total_count = len(ious)
        count_iou_025 = (np.array(ious) > 0.25).sum()
        count_iou_05 = (np.array(ious) > 0.5).sum()

        # mIoU
        average_iou = np.mean(ious)
        f.write(f"Average IoU: {average_iou:.4f}\n")
        f.write(f"Acc@0.25: {count_iou_025 / total_count:.4f}\n")
        f.write(f"Acc@0.5: {count_iou_05 / total_count:.4f}\n")


if __name__ == "__main__":
    parser = ArgumentParser("Compute LeRF IoU")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    # compulsaory
    parser.add_argument("--output_folder", type=str, default="output", help="Output folder for evaluation results")
    args = parser.parse_args()
    if not args.scene_name:
        parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")
    if not args.output_folder:
        parser.error("The --output_folder argument is required")

    # TODO: change
    path_gt = f"/media/tianyu/hard_drive/Dataset/OpenGaussian/langsplat/lerf/label/{args.scene_name}/gt"
    # renders_cluster_silhouette is the predicted mask
    path_pred = f"{args.output_folder}/{args.scene_name}/text2obj/ours_70001/renders_cluster_silhouette"
    output_file = f"{args.output_folder}/{args.scene_name}/evaluation_results.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    evalute(path_gt, path_pred, args.scene_name, output_file=output_file)