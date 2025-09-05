#!/bin/bash
# create_training_config.sh - SAM2.1学習設定ファイルを生成

set -eu

echo "SAM2.1学習設定ファイルを作成します..."

# プロジェクトルートを取得
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 設定ファイルのパス
CONFIG_DIR="configs/sam2.1_training"
CONFIG_FILE="$CONFIG_DIR/sam2.1_hiera_b+_foodmix_finetune.yaml"

# ディレクトリ作成
mkdir -p "$CONFIG_DIR"

# GPUメモリに基づくバッチサイズ推定
BATCH_SIZE=1
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -ge 24000 ]; then
        BATCH_SIZE=2
    elif [ "$GPU_MEM" -ge 40000 ]; then
        BATCH_SIZE=4
    fi
fi

# データパスの絶対パス取得
DATA_DIR="$PROJECT_ROOT/data/foodmix_sa1b"
SAM2_DIR="$PROJECT_ROOT/external/sam2"

# 既存の設定ファイルがある場合はバックアップ
if [ -f "$CONFIG_FILE" ]; then
    BACKUP_FILE="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "既存の設定ファイルをバックアップ: $BACKUP_FILE"
    cp "$CONFIG_FILE" "$BACKUP_FILE"
fi

# 簡略版の設定ファイルを生成（SAM2の実際の構造に合わせて調整）
cat > "$CONFIG_FILE" << EOF
# SAM2.1 Fine-tuning Configuration for Food Dataset
# Auto-generated on $(date)

# Paths (relative to sam2 directory)
experiment_log_dir: ./sam2_logs/foodmix_finetune

# Model
model:
  checkpoint: ./checkpoints/sam2.1_hiera_base_plus.pt
  model_cfg: configs/sam2.1/sam2.1_hiera_b+.yaml

# Dataset
data:
  train:
    batch_sizes: [$BATCH_SIZE]
    img_folder: $DATA_DIR/images
    gt_folder: $DATA_DIR/annotations
    file_list_txt: $DATA_DIR/train.txt
    num_frames: 1
    max_num_objects: 50
    
  val:
    img_folder: $DATA_DIR/images
    gt_folder: $DATA_DIR/annotations
    file_list_txt: $DATA_DIR/val.txt
    num_frames: 1
    max_num_objects: 50

# Training parameters
trainer:
  max_epochs: 40
  num_workers: 4
  precision: bf16  # Change to fp32 for older GPUs
  grad_accum_steps: 1

# Optimizer
optimizer:
  name: AdamW
  lr: 1.0e-4
  weight_decay: 0.05
  betas: [0.9, 0.999]

# Checkpointing
checkpoint:
  save_freq: 5  # Save every N epochs
  keep_last_k: 3

# Random seed
seed: 42
EOF

echo "設定ファイルを作成しました: $CONFIG_FILE"
echo ""
echo "=== 設定内容 ==="
echo "バッチサイズ: $BATCH_SIZE"
echo "データディレクトリ: $DATA_DIR"
echo "チェックポイント: sam2.1_hiera_base_plus.pt"
echo "エポック数: 40"
echo ""
echo "必要に応じて $CONFIG_FILE を編集してください"