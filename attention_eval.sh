cd /home/ahmed_heakl/pts
source ~/.bashrc
conda activate ahmed

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=4,5,6,7
for name_architecture in "diffusion-llm"; do
    for src_subset in "arc_easy" "arc_challenge" "dart-1" "dart-2" "dart-3" "dart-4" "dart-5" "gsm8k" "mmlu" "aime" ; do
        python pts/eval/eval_attention_parallel.py \
            --config configs/attention.yaml \
            --dataset $src_subset \
            --num_samples 200 \
            --name_architecture "$name_architecture" \
            --attention True
    done
done


# for name_architecture in "llm-only" "diffusion-llm"; do
#     for src_subset in "truthfulqa" ; do
#         python pts/eval/eval_math_parallel.py \
#             --config configs/default.yaml \
#             --dataset $src_subset \
#             --num_samples 200 \
#             --name_architecture "$name_architecture"
#     done
# done




