source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a

python -m pts.cli \
    --config configs/default.yaml \
    --no-refined \ 
    --number_samples 20