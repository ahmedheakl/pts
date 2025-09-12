cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate .heakl
module load nvidia/cuda/11.8

set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=2,3
python pts/train/train.py


