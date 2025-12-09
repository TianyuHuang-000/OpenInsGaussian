scan="duola"
gpu_num=0
CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
    -s /home/tianyu/Documents/GitHub/OpenGaussian_v0/data/${scan} \
    -r 2 \
    --iterations 70_000 \
    --start_ins_feat_iter 30_000 \
    --start_root_cb_iter 40_000 \
    --start_leaf_cb_iter 50_000 \
    --sam_level 0 \
    --root_node_num 64 \
    --leaf_node_num 5 \
    --pos_weight 1.0 \
    --test_iterations 30000 \
    --eval \
    --model_path /home/tianyu/Documents/GitHub/OpenGaussian_v0/output/${scan}