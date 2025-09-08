#!/bin/bash
# create_training_config_large.sh - SAM2.1 Hiera-Large学習設定ファイルを生成

set -eu

echo "SAM2.1 Hiera-Large学習設定ファイルを作成します..."

# プロジェクトルートを取得
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 設定ファイルのパス - Largeモデル用
CONFIG_DIR="configs/sam2.1_training"
CONFIG_FILE="$CONFIG_DIR/sam2.1_hiera_l_foodmix_finetune.yaml"

# ディレクトリ作成
mkdir -p "$CONFIG_DIR"

# Largeモデル用：より厳格なメモリ判定
BATCH_SIZE=1  # Largeモデルは基本的に1に固定
MAX_NUM_OBJECTS=30  # Base+の50から30に削減
NUM_WORKERS=2  # メモリ効率化

if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU メモリ: ${GPU_MEM}MB"
    
    if [ "$GPU_MEM" -lt 20000 ]; then
        echo "警告: Largeモデルには20GB以上のGPUメモリを推奨します"
        echo "メモリ最適化設定を適用します..."
        MAX_NUM_OBJECTS=20
        NUM_WORKERS=1
    elif [ "$GPU_MEM" -ge 32000 ]; then
        echo "充分なGPUメモリが利用可能です"
        MAX_NUM_OBJECTS=30
    fi
else
    echo "警告: nvidia-smiが見つかりません。保守的な設定を使用します"
    MAX_NUM_OBJECTS=20
    NUM_WORKERS=1
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

# Largeモデル用設定ファイルを生成
cat > "$CONFIG_FILE" << EOF
# @package _global_
# SAM2.1 Hiera-Large Food Dataset Fine-tuning Configuration
# Auto-generated on $(date)

# Default base configuration to inherit
defaults:
  - default

# Experiment settings - Large model specific
experiment_log_dir: ./sam2_logs/foodmix_l_finetune

# Dataset configuration
dataset:
  img_folder: \${hydra:runtime.cwd}/data/foodmix_sa1b/images
  gt_folder: \${hydra:runtime.cwd}/data/foodmix_sa1b/annotations
  file_list_txt: \${hydra:runtime.cwd}/data/foodmix_sa1b/train.txt

# Model configuration - CHANGED: Base+ -> Large
model:
  checkpoint: \${hydra:runtime.cwd}/external/sam2/checkpoints/sam2.1_hiera_large.pt
  model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml

# Training configuration - Optimized for Large model
scratch:
  train_batch_size: $BATCH_SIZE  # Fixed for Large model memory requirements
  num_epochs: 40
  max_num_objects: $MAX_NUM_OBJECTS  # Reduced for Large model memory optimization
  num_train_workers: $NUM_WORKERS  # Reduced worker count for memory efficiency

# Optimizer configuration - Large model optimized
optimizer:
  base_lr: 1.0e-4  # Standard learning rate within SAM2.1 recommended range
  vision_lr: 3.0e-6  # Lower learning rate for Vision Encoder (Hiera)
  weight_decay: 0.05

# Mixed precision training for Large model efficiency
amp:
  enabled: true
  amp_dtype: bfloat16  # bf16 recommended for A100/H100, more stable than fp16

# Memory optimization settings
memory:
  pin_memory: false  # Disable to prevent memory leaks
  gradient_accumulation_steps: 1  # Can be increased if batch_size needs to be reduced further

# Validation dataset
val_dataset:
  img_folder: \${hydra:runtime.cwd}/data/foodmix_sa1b/images
  gt_folder: \${hydra:runtime.cwd}/data/foodmix_sa1b/annotations
  file_list_txt: \${hydra:runtime.cwd}/data/foodmix_sa1b/val.txt

# Hydra configuration
hydra:
  run:
    dir: \${experiment_log_dir}
  sweep:
    dir: \${experiment_log_dir}
    subdir: \${hydra.job.num}
EOF

echo "Largeモデル用設定ファイルを作成しました: $CONFIG_FILE"
echo ""
echo "=== Largeモデル設定内容 ==="
echo "モデル: SAM2.1 Hiera-Large"
echo "チェックポイント: sam2.1_hiera_large.pt"
echo "バッチサイズ: $BATCH_SIZE (Largeモデル固定)"
echo "最大オブジェクト数: $MAX_NUM_OBJECTS"
echo "ワーカー数: $NUM_WORKERS"
echo "データディレクトリ: $DATA_DIR"
echo "エポック数: 40"
echo "混合精度: bfloat16"
echo ""
echo "必要に応じて $CONFIG_FILE を編集してください"
echo ""
echo "学習開始方法:"
echo "  bash scripts/20_train_food_sam2_large.sh"