source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=0,1,2,3


for name_architecture in "diffusion-llm" "diffusion-only" "llm-diffusion" "llm-only"; do
    python -m pts.cli \
    --config configs/default.yaml \
    --no-refine \
    --number_samples 20 \
    --name_architecture "$name_architecture"
done