#!/bin/bash
#SBATCH -J big
#SBATCH --qos large_qos
#SBATCH --gres gpu:2
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-cpu 4196
#SBATCH -o slurm/job_%j.log
#SBATCH -e slurm/job_%j.log

use_port=9911

while [[ $use_port -lt $(($use_port + 100)) ]]; do
    if ! lsof -i :$use_port -sTCP:LISTEN > /dev/null 2>&1; then
        echo "Port $use_port is available. Launching torchrun..."
        break
    fi
    ((use_port++))
done

HF_HUB_OFFLINE=1 torchrun --master_port=$use_port --nproc_per_node=2 tools/train.py \
 -c configs/rtdetr/bsr5detr_512_ruod.yml \
 --amp \
 --seed=0