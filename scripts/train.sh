# Total training steps: 70k
# 3dgs pre-train: 0~30k
# stage1: 30~40k
# stage2 (coarse-level): 40~50k
# stage2 (fine-level): 50k~70k
dataset="combined"
scan="ramen"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
    -s "/media/tianyu/My Passport/Data/OpenGaussian_clip_base_aug/${dataset}/lerf/${scan}" \
    --iterations 70_001 \
    --start_ins_feat_iter 30_000 \
    --start_root_cb_iter 40_000 \
    --start_leaf_cb_iter 50_000 \
    --sam_level 3 \
    --root_node_num 64 \
    --leaf_node_num 10 \
    --pos_weight 0.5 \
    --loss_weight 0.01 \
    --test_iterations 30000 \
    --eval \
    --model_path "/home/tianyu/Documents/GitHub/OpenInsGaussian/output/lerf/${scan}"