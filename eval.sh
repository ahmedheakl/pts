source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=2,3
# python pts/eval/eval_arc.py \
#     --num_samples 100

python pts/eval/eval_arc_parallel.py \
    --config configs/default.yaml \
    --dataset allenai/ai2_arc \
    --subset ARC-Challenge \
    --split test \
    --num_samples 4
