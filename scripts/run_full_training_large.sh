#!/bin/bash

# SAM2.1 Hiera-Large フルスケール学習（40エポック）を実行

set -e

echo "============================================"
echo "SAM2.1 Hiera-Large Food Fine-tuning - Full Scale (40 epochs)"
echo "============================================"

# ログディレクトリの作成
LOG_DIR="/home/soya/sam2_1_food_finetuning/training_logs"
mkdir -p $LOG_DIR

# 現在時刻を取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_large_40epochs_$TIMESTAMP.log"

echo "ログファイル: $LOG_FILE"
echo ""

# GPU情報を表示と十分性チェック
echo "GPU情報とLargeモデル対応チェック:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader
    
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEMORY" -lt 20000 ]; then
        echo ""
        echo "⚠️  警告: Largeモデルには20GB以上のGPUメモリを推奨します (現在: ${GPU_MEMORY}MB)"
        echo "メモリ最適化版の使用を検討してください:"
        echo "  bash scripts/train_with_monitoring.sh --large-model --memory-optimized"
        echo ""
        read -p "続行しますか？ (y/N): " CONTINUE
        if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
            echo "学習を中止しました"
            exit 0
        fi
    else
        echo "✅ GPUメモリ充分: Largeモデル学習に適しています"
    fi
else
    echo "GPU情報取得失敗 - nvidia-smi が見つかりません"
    exit 1
fi
echo ""

echo "Largeモデル特性:"
echo "  - パラメータ数: ~224M (Base+の約4.5倍)"
echo "  - 計算時間: Base+の1.5-2倍"
echo "  - メモリ使用量: 大幅増加"
echo "  - バッチサイズ: 1固定"
echo ""

# nohupでバックグラウンド実行
echo "Largeモデル学習をバックグラウンドで開始します..."
echo "進捗確認: tail -f $LOG_FILE"
echo ""

cd /home/soya/sam2_1_food_finetuning

# Largeモデル用設定を使用
nohup python external/sam2/training/train.py \
    -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_40epochs.yaml \
    --use-cluster 0 \
    --num-gpus 1 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "プロセスID: $PID"
echo ""

# pidファイルに保存
echo $PID > "$LOG_DIR/training_large_40epochs.pid"

echo "Largeモデル学習が開始されました。"
echo ""
echo "状態確認コマンド:"
echo "  進捗確認: tail -f $LOG_FILE"
echo "  プロセス確認: ps -p $PID"
echo "  停止: kill $PID"
echo "  GPU使用状況: nvidia-smi"
echo "  TensorBoard: tensorboard --logdir /home/soya/sam2_1_food_finetuning/external/sam2/sam2_logs/foodmix_l_40epochs_full"
echo ""
echo "Largeモデル固有の注意事項:"
echo "  - 学習時間はBase+より長くなります"
echo "  - メモリ使用量を定期的に監視してください"
echo "  - OOM発生時は monitor_system でリソース使用状況を確認"
echo ""
echo "============================================"