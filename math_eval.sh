cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
for name_architecture in "diffusion-diffusion" "diffusion-llm" "llm-diffusion" "diffusion-only" "llm-only" "llm-llm"; do
    for src_subset in "mmlu"; do
        python pts/eval/eval_arc_parallel.py \
            --config configs/llada_llama.yaml \
            --dataset $src_subset \
            --num_samples 200 \
            --name_architecture "$name_architecture"
    done
done
