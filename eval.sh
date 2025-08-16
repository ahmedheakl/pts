source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# for arc_subset in "ARC-Easy" "ARC-Challenge"; do
#     python pts/eval/eval_arc_parallel.py \
#         --config configs/default.yaml \
#         --dataset allenai/ai2_arc \
#         --subset $arc_subset \
#         --split test \
#         --num_samples 200
# done


export CUDA_VISIBLE_DEVICES=0
for arc_subset in "ARC-Easy" "ARC-Challenge"; do
    python pts/eval/eval_arc.py \
        --config configs/default.yaml \
        --dataset allenai/ai2_arc \
        --subset $arc_subset \
        --split test \
        --num_samples 200
done
