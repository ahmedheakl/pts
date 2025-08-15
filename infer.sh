source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

python -m pts.cli \
    --config configs/default.yaml \
    --prompt "Write a plan for writing an essay about the pros and cons of artificial intelligence in education. " \
    --extra "Write an essay following this plan" \
    --no-refine