cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0
for name_architecture in "diffusion-diffusion" "llm-llm" "llm-only" "diffusion-only"; do
    for src_subset in "ielts-essays"; do
        python pts/eval/eval_essay_parallel.py \
            --config configs/default.yaml \
            --dataset $src_subset \
            --num_samples 4 \
            --name_architecture "$name_architecture"
    done
done
