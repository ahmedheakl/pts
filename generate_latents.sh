#!/bin/bash
cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

datasets=("arc_easy" "arc_challenge" "dart-1" "dart-2")
# datasets=("dart-3" "dart-4" "dart-5" "gsm8k")

# number of GPUs
ngpus=4

for i in "${!datasets[@]}"; do
    gpu=$((i % ngpus))
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "Launching ${datasets[$i]} on GPU $gpu"
    python pts/train/generate_latents.py \
        --num_samples 1000 \
        --dataset "${datasets[$i]}" > logs/${datasets[$i]}.log 2>&1 &
done

wait
echo "All jobs finished."
