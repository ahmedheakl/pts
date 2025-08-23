cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
for name_architecture in "llm-diffusion" "diffusion-llm"; do
    for src_subset in "ielts-essays"; do
        python pts/eval/eval_essay_parallel.py \
            --config configs/default.yaml \
            --dataset $src_subset \
            --num_samples 200 \
            --name_architecture "$name_architecture"
    done
done
