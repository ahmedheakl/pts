source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a


export CUDA_VISIBLE_DEVICES=0,1,2,3
for src_subset in "gsm8k" ; do
    python pts/eval/eval_arc_parallel.py \
        --config configs/default.yaml \
        --dataset $src_subset \
        --num_samples 200 \
        --architecture_name "diffusion-llm" \
        
done

