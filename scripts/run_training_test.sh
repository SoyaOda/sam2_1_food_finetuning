#!/bin/bash

# SAM2.1の学習を実行するスクリプト（テスト版）

set -e

echo "============================================"
echo "SAM2.1 Food Fine-tuning (Test Run)"
echo "============================================"

# SAM2ディレクトリに移動
cd /home/soya/sam2_1_food_finetuning/external/sam2

# 現在のディレクトリを確認
echo "Current directory: $(pwd)"

# データパスを絶対パスで設定
DATA_DIR="/home/soya/sam2_1_food_finetuning/data/foodmix_sa1b"

# 設定ファイルの存在確認
CONFIG_FILE="sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_test.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo "Data directory: $DATA_DIR"

# Pythonパスを設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Hydraのデバッグを有効化
export HYDRA_FULL_ERROR=1

# GPUメモリの確認
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader || echo "No NVIDIA GPU detected"

echo ""
echo "Starting training..."
echo ""

# 実行コマンド（オーバーライドを使って設定を直接指定）
python training/train.py \
    -c sam2.1_training/sam2.1_hiera_b+_foodmix_test \
    --use-cluster 0 \
    --num-gpus 1 \
    ++dataset.img_folder="${DATA_DIR}/images" \
    ++dataset.gt_folder="${DATA_DIR}/annotations" \
    ++dataset.file_list_txt="${DATA_DIR}/train.txt" \
    ++trainer.max_epochs=1 \
    ++launcher.experiment_log_dir="./sam2_logs/test_run_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "============================================"
echo "Training completed (or failed - check logs)"
echo "============================================"