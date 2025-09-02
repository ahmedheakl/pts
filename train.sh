cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate arina-cass

set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=0,1
python pts/train/train.py