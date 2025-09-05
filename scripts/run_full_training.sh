#!/bin/bash

# SAM2.1 フルスケール学習（40エポック）を実行

set -e

echo "============================================"
echo "SAM2.1 Food Fine-tuning - Full Scale (40 epochs)"
echo "============================================"

# ログディレクトリの作成
LOG_DIR="/home/soya/sam2_1_food_finetuning/training_logs"
mkdir -p $LOG_DIR

# 現在時刻を取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_40epochs_$TIMESTAMP.log"

echo "ログファイル: $LOG_FILE"
echo ""

# GPU情報を表示
echo "GPU情報:"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader || echo "GPU情報取得失敗"
echo ""

# nohupでバックグラウンド実行
echo "学習をバックグラウンドで開始します..."
echo "進捗確認: tail -f $LOG_FILE"
echo ""

cd /home/soya/sam2_1_food_finetuning

nohup python scripts/train_sam2_wrapper.py \
    -c sam2.1_training/sam2.1_hiera_b+_foodmix_40epochs \
    --use-cluster 0 \
    --num-gpus 1 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "プロセスID: $PID"
echo ""

# pidファイルに保存
echo $PID > "$LOG_DIR/training_40epochs.pid"

echo "学習が開始されました。"
echo ""
echo "状態確認コマンド:"
echo "  進捗確認: tail -f $LOG_FILE"
echo "  プロセス確認: ps -p $PID"
echo "  停止: kill $PID"
echo "  GPU使用状況: nvidia-smi"
echo "  TensorBoard: tensorboard --logdir /home/soya/sam2_1_food_finetuning/external/sam2/sam2_logs/foodmix_40epochs_full"
echo ""
echo "============================================"