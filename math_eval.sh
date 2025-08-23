source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
for name_architecture in "diffusion-llm" "diffusion-only" "llm-diffusion" "llm-only"; do
    for src_subset in "arc_easy" "arc_challenge" "dart-1" "dart-2" "dart-3" "dart-4" "dart-5" "gsm8k"; do
        python pts/eval/eval_arc_parallel.py \
            --config configs/default.yaml \
            --dataset $src_subset \
            --num_samples 4 \
            --name_architecture "$name_architecture"
    done
done
