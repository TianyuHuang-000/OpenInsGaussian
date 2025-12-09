scene_list=("ramen" "teatime" "waldo_kitchen" "figurines")
# scene_list=("figurines")

model="/home/tianyu/Documents/GitHub/OpenInsGaussian/output/rerun/lerf"
for scene in "${scene_list[@]}"; do
    python render_lerf_by_text.py -m "${model}/${scene}" --scene_name "${scene}"
    python scripts/compute_lerf_iou.py --scene_name "${scene}" --output_folder "${model}"
done




