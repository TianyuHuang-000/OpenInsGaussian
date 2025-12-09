import os
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import numpy as np
import torch
import json

nyu40_dict = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}

# ScanNet 20 classes
scannet19_dict = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 14: "desk", 16: "curtain",
    24: "refrigerator", 28: "shower curtain", 33: "toilet", 34: "sink",
    36: "bathtub", # 39: "otherfurniture"
}

import numpy as np  
def sigmoid(x):  
    return 1 / (1 + np.exp(-x))  

def write_ply(vertex_data, output_path):
    vertices = []
    for vertex in vertex_data:
        r = (vertex['ins_feat_r'] + 1)/2 * 255
        g = (vertex['ins_feat_g'] + 1)/2 * 255
        b = (vertex['ins_feat_b'] + 1)/2 * 255
        new_vertex = (vertex['x'], vertex['y'], vertex['z'], r, g, b)
        vertices.append(new_vertex)
    
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertex_data = np.array(vertices, dtype=vertex_dtype)
    
    el = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([el], text=True).write(output_path)

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    # Extract the coordinates and labels of the points. The labels are from 1 to 40 for the NYU40 dataset, with 0 being invalid.
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels

def calculate_metrics(gt, pred, total_classes):
    print(gt.shape, pred.shape, total_classes)

    gt = gt.cpu()
    pred = pred.cpu()
    pred[gt == 0] = 0



    ious = torch.zeros(total_classes)

    intersection = torch.zeros(total_classes)
    union = torch.zeros(total_classes)
    correct = torch.zeros(total_classes)
    total = torch.zeros(total_classes)

    for cls in range(1, total_classes):
        intersection[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        union[cls] = torch.sum((gt == cls) | (pred == cls)).item()
        correct[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        total[cls] = torch.sum(gt == cls).item()

    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    # Only consider the categories that exist in the current scene
    gt_classes = torch.unique(gt)
    valid_gt_classes = gt_classes[gt_classes != 0]  # ignore 0

    # miou
    mean_iou = ious[valid_gt_classes].mean().item()

    # acc
    valid_mask = gt != 0
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else float('nan')

    class_accuracy = correct / total
    # mAcc.
    mean_class_accuracy = class_accuracy[valid_gt_classes].mean().item()

    return ious, mean_iou, accuracy, mean_class_accuracy

def generate_label_masks(labels):
    """
    Generate a mask matrix for unique labels.

    Parameters:
    labels (list or np.ndarray): A 1D array-like object with int labels.

    Returns:
    np.ndarray: A 2D array of shape where each column is a mask
                for one unique label.
    """
    # Convert input to numpy array if it is not already
    labels = np.asarray(labels)

    # Get the unique labels and the number of unique labels
    unique_labels = np.unique(labels)
    n_unique = len(unique_labels)

    # Initialize the mask matrix
    masks = np.zeros((len(labels), n_unique), dtype=int)

    # Populate the mask matrix
    for i, unique_label in enumerate(unique_labels):
        masks[:, i] = (labels == unique_label).astype(int)

    return masks
    
if __name__ == "__main__":
    scene_list = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00", "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
    # scene_list = [ "scene0000_00"]

    # Iterations and other setup
iteration = 90001

# Define different target class configurations
target_configs = {
    '19_classes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    '15_classes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],
    '10_classes': [1, 2, 4, 5, 6, 7, 8, 9, 10, 33]
}

# Main processing loop
for config_name, target_id in target_configs.items():
    # Update target dictionary and names
    target_dict = {key: nyu40_dict[key] for key in target_id}
    target_names = list(target_dict.values())
    miou = 0.0
    macc = 0.0

    for scan_name in scene_list:
        # (1) GT ply
        gt_file_path = f"/media/tianyu/hard_drive/Dataset/OpenGaussian/langsplat/scannet/{scan_name}/{scan_name}_vh_clean_2.labels.ply"
        points, labels = read_labels_from_ply(gt_file_path)

        # (3) Update gt label
        target_id_mapping = {value: index + 1 for index, value in enumerate(target_id)}
        updated_labels = np.zeros_like(labels)
        for original_value, new_value in target_id_mapping.items():
            updated_labels[labels == original_value] = new_value
        updated_gt_labels = torch.from_numpy(updated_labels.astype(np.int64)).cuda()

        # (4) Load Gaussian ply file
        model_name = '/home/tianyu/Documents/GitHub/OpenInsGaussian/output/rerun/scannet'
        model_path = f"{model_name}/{scan_name}/"
        ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex'].data
        ignored_pts = sigmoid(vertex_data["opacity"]) < 0.1
        updated_gt_labels[ignored_pts] = 0

        # (5) Load cluster language file
        mapping_file = os.path.join(model_path, "cluster_lang.npz")
        saved_data = np.load(mapping_file)
        leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()
        
    
        leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()
        leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()
        leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()
        leaf_lang_feat[leaf_occu_count < 2] *= 0.0
        leaf_ind = leaf_ind.clamp(max=319)

        # (6) Load query text features
        feature_len = leaf_lang_feat.shape[1]
        with open(f'assets/text_features_oig.json', 'r') as f:
            data_loaded = json.load(f)
        all_texts = list(data_loaded.keys())
        text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)
        
        query_text_feats = torch.zeros(len(target_names), feature_len).cuda()
        for i, text in enumerate(target_names):
            feat = text_features[all_texts.index(text)].unsqueeze(0)
            query_text_feats[i] = feat

        # (7) Calculate cosine similarity
        query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
        leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)
        cosine_similarity = torch.matmul(query_text_feats, leaf_lang_feat.transpose(0, 1))
        max_id = torch.argmax(cosine_similarity, dim=0)
        pred_pts_cls_id = max_id[leaf_ind] + 1


        # torch.save(updated_gt_labels, f'/home/tianyu/Documents/GitHub/OpenGaussian_v0/output/test/{scan_name}_gt_19.pth')
        # torch.save(pred_pts_cls_id, f'/home/tianyu/Documents/GitHub/OpenGaussian_v0/output/test/{scan_name}_pred_10.pth')
        # copy the gt file to the output folder
        # os.system(f"cp {gt_file_path} /home/tianyu/Documents/GitHub/OpenGaussian_v0/output/test/{scan_name}.ply")

        # Evaluate and write results to specific config file
        ious, mean_iou, accuracy, mean_acc = calculate_metrics(updated_gt_labels, pred_pts_cls_id, total_classes=len(target_names)+1)
        results_file = f"{model_name}/evaluation_{config_name}.txt"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, "a") as f:
            f.write(f"Scene: {scan_name}, mIoU: {mean_iou:.4f}, mAcc.: {mean_acc:.4f}\n")
        print(f"Scene: {scan_name}, {config_name}, mIoU: {mean_iou:.4f}, mAcc.: {mean_acc:.4f}")
        miou += mean_iou
        macc += mean_acc

    miou /= len(scene_list)
    macc /= len(scene_list)
    results_file = f"{model_name}/evaluation_{config_name}.txt"
    with open(results_file, "a") as f:
        f.write(f"Scene: Mean, mIoU: {miou:.4f}, mAcc.: {macc:.4f}\n")
