#!/bin/bash
# chmod +x scripts/train_scannet.sh
# ./scripts/train_scannet.sh

# ============== [Notice] ==============
# 1. The 10 scene hyperparameters in the ScanNet dataset are consistent.
# 2. Train a scene for about 20 minutes on a 24G 4090 GPU.
# 3. Please check the dataset path specified by -s.

# ============== [Hyperparameter explanation] ==============
# Total training steps: 9w
# 3dgs pre-train: 0~3w
# stage1: 3~5w
# stage2 (coarse-level): 5~7w
# stage2 (fine-level): 7~9w
# k1=64, k2=5
# frozen_init_pts: The point clouds provided by the ScanNet dataset are frozen, without using the densification scheme of 3DGS.
# -r 2 : We use half-resolution data for training.

# ============== [10 scenes] ==============
scan_list=("scene0000_00" "scene0062_00" "scene0070_00" "scene0097_00" "scene0140_00" \
"scene0200_00" "scene0347_00" "scene0400_00" "scene0590_00" "scene0645_00")
# scan_list=("scene0000_00")
gpu_num=0

for scan in "${scan_list[@]}"; do
    echo "Training for ${scan} ....."
    log_dir="./logs/${scan}"
    mkdir -p "$log_dir"

    echo "Starting training for $scan..." | tee "$log_dir/train.log"

    # Record memory usage every 5s in background
    nvidia-smi --query-gpu=timestamp,index,name,memory.used --format=csv -l 5 > "$log_dir/mem.log" &
    mem_logger_pid=$!

    # Record time and stdout+stderr
    { time CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
        -s "/home/tianyu/Documents/GitHub/OpenInsGaussian/data/rerun/scannet/${scan}" \
        -r 2 \
        --frozen_init_pts \
        --iterations 90_001 \
        --start_ins_feat_iter 30_000 \
        --start_root_cb_iter 50_000 \
        --start_leaf_cb_iter 70_000 \
        --sam_level 0 \
        --root_node_num 64 \
        --leaf_node_num 5 \
        --pos_weight 1.0 \
        --test_iterations 30000 \
        --eval \
        --model_path "/home/tianyu/Documents/GitHub/OpenInsGaussian/output/rerun/scannet/${scan}" \
        --start_checkpoint /media/tianyu/hard_drive/output/OpenGaussian/scannet_ori_ori/${scan}/chkpnt90000.pth ; } \
        2>&1 | tee -a "$log_dir/train.log"

    # Stop memory logger
    kill $mem_logger_pid
done
