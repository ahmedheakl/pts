cd /home/abdulrahman.mahmoud/HEAKL/PTS
source ~/.bashrc
conda activate .heakl
module load nvidia/cuda/11.8

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=1
python pts/train/generate_latents.py --dataset "dart-5" --num_samples 5000

# for dataset in "arc_challenge" "dart-1" "dart-2" "dart-3" "dart-4" "dart-5" "gsm8k":
# do
export CUDA_VISIBLE_DEVICES=0
python pts/train/generate_latents.py --dataset "dart-4" --num_samples 5000


export CUDA_VISIBLE_DEVICES=1
python pts/train/generate_latents.py --dataset "dart-5" --num_samples 5000

export CUDA_VISIBLE_DEVICES=1
python pts/train/generate_latents.py --dataset "dart-2" --num_samples 5000


export CUDA_VISIBLE_DEVICES=3
python pts/train/generate_latents.py --dataset "dart-3" --num_samples 5000