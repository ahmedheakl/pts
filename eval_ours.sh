cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate .heakl

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
for name_architecture in "dl_dual"; do
    for src_subset in "aime" ; do
        python pts/eval/eval_ours.py \
            --config configs/dual_pipeline.yaml \
            --dataset $src_subset \
            --num_samples 200 \
            --name_architecture "$name_architecture"
    done
done