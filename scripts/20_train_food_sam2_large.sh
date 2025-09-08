#!/bin/bash
# 20_train_food_sam2_large.sh - SAM2.1 Hiera-Large モデルの学習実行スクリプト
set -eux

echo "SAM2.1 Hiera-Large Food Finetuning 学習を開始します..."

# プロジェクトルートを取得
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 必要なディレクトリとファイルの存在確認
echo "環境チェック中..."

# SAM2ディレクトリ確認
if [ ! -d "external/sam2" ]; then
    echo "エラー: external/sam2 ディレクトリが見つかりません"
    echo "先に bash scripts/00_setup_env.sh を実行してください"
    exit 1
fi

# データディレクトリ確認
if [ ! -d "data/foodmix_sa1b/images" ] || [ ! -d "data/foodmix_sa1b/annotations" ]; then
    echo "エラー: data/foodmix_sa1b にデータが見つかりません"
    echo "前処理スクリプトを実行してください:"
    echo "  python scripts/10_prepare_foodseg103.py"
    echo "  python scripts/11_prepare_uecfoodpix.py"
    echo "  python scripts/12_merge_to_sa1b.py"
    exit 1
fi

# train.txt/val.txt確認
if [ ! -f "data/foodmix_sa1b/train.txt" ] || [ ! -f "data/foodmix_sa1b/val.txt" ]; then
    echo "エラー: train.txt/val.txt が見つかりません"
    echo "python scripts/12_merge_to_sa1b.py を実行してください"
    exit 1
fi

# Largeモデル用チェックポイント確認
LARGE_CHECKPOINT="external/sam2/checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "$LARGE_CHECKPOINT" ]; then
    echo "エラー: Largeモデルのチェックポイントが見つかりません: $LARGE_CHECKPOINT"
    echo "以下のコマンドでダウンロードしてください:"
    echo "  cd external/sam2 && bash checkpoints/download_ckpts.sh"
    exit 1
fi

# 設定ファイルの存在確認
CONFIG_FILE="configs/sam2.1_training/sam2.1_hiera_l_foodmix_finetune.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "エラー: Largeモデル用設定ファイルが見つかりません: $CONFIG_FILE"
    echo "設定ファイルが作成されているか確認してください"
    exit 1
fi

# GPU確認とメモリ最適化の判定
echo ""
echo "=== GPU情報 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "検出されたGPU数: $GPU_COUNT"
    
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU メモリ: ${GPU_MEM}MB"
    
    # Largeモデル用のメモリ判定
    if [ "$GPU_MEM" -lt 20000 ]; then
        echo ""
        echo "警告: Largeモデルには20GB以上のGPUメモリを推奨します"
        echo "メモリ最適化版設定の使用を検討してください:"
        echo "  bash scripts/train_with_monitoring.sh --config sam2.1_training/sam2.1_hiera_l_foodmix_optimized --memory-optimized"
        echo ""
        read -p "メモリ最適化版を使用しますか？ (y/n): " USE_OPTIMIZED
        if [ "$USE_OPTIMIZED" = "y" ]; then
            echo "メモリ最適化版での学習に切り替えます..."
            exec bash scripts/train_with_monitoring.sh --config sam2.1_training/sam2.1_hiera_l_foodmix_optimized --memory-optimized
        fi
    fi
else
    echo "警告: nvidia-smiが見つかりません。CPU環境で実行されます"
    GPU_COUNT=0
fi

# データ統計表示
echo ""
echo "=== データセット統計 ==="
TRAIN_COUNT=$(wc -l < data/foodmix_sa1b/train.txt)
VAL_COUNT=$(wc -l < data/foodmix_sa1b/val.txt)
echo "訓練データ: $TRAIN_COUNT ファイル"
echo "検証データ: $VAL_COUNT ファイル"

# 学習パラメータ設定
NUM_GPUS=${NUM_GPUS:-$GPU_COUNT}
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1  # CPU実行の場合
fi

# Largeモデル固有の警告
echo ""
echo "=== Largeモデル固有の注意事項 ==="
echo "- Largeモデルは Base+ より1.5-2倍の計算時間がかかります"
echo "- バッチサイズは1に固定されています（メモリ最適化のため）"
echo "- max_num_objects は30に設定されています（Base+: 50 → Large: 30）"
echo "- チェックポイントは250ステップごとに保存されます"

# SAM2ディレクトリに移動
cd external/sam2

# 学習コマンド構築
TRAIN_CMD="python training/train.py"
TRAIN_CMD="$TRAIN_CMD -c ../../$CONFIG_FILE"
TRAIN_CMD="$TRAIN_CMD --use-cluster 0"
TRAIN_CMD="$TRAIN_CMD --num-gpus $NUM_GPUS"

# 追加オプション（必要に応じて）
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CHECKPOINT"
fi

# 実行確認
echo ""
echo "=== Largeモデル学習コマンド ==="
echo "$TRAIN_CMD"
echo ""
echo "設定:"
echo "  - モデル: SAM2.1 Hiera-Large"
echo "  - GPU数: $NUM_GPUS"
echo "  - バッチサイズ: 1（Largeモデル固定）"
echo "  - max_num_objects: 30"
echo "  - 訓練データ: $TRAIN_COUNT"
echo "  - 検証データ: $VAL_COUNT"
echo "  - 混合精度: bfloat16"

echo ""
read -p "Largeモデルの学習を開始しますか？ (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "学習を中止しました"
    exit 0
fi

# 学習実行
echo ""
echo "SAM2.1 Hiera-Large モデルの学習を開始します..."
echo "ログは external/sam2/sam2_logs/foodmix_l_finetune/ に保存されます"
echo "Ctrl+C で中断できます（チェックポイントから再開可能）"
echo ""

# 学習実行（エラーハンドリング付き）
set +e
$TRAIN_CMD
TRAIN_EXIT_CODE=$?
set -e

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Largeモデルの学習が正常に完了しました！ ==="
    echo "チェックポイント: external/sam2/sam2_logs/foodmix_l_finetune/checkpoints/"
    echo ""
    echo "次のステップ:"
    echo "1. python scripts/21_eval_and_viz.py --config external/sam2/configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint <学習済みチェックポイント>"
    echo "2. python scripts/visualize_sam2_predictions.py --model-cfg sam2_hiera_l.yaml --checkpoint <学習済みチェックポイント>"
else
    echo ""
    echo "=== Largeモデルの学習中にエラーが発生しました ==="
    echo "終了コード: $TRAIN_EXIT_CODE"
    echo ""
    echo "Largeモデル固有のトラブルシューティング:"
    echo "1. メモリ不足の場合:"
    echo "   - メモリ最適化版を使用: bash scripts/train_with_monitoring.sh --config sam2.1_training/sam2.1_hiera_l_foodmix_optimized --memory-optimized"
    echo "   - max_num_objects を 25 → 15 → 10 → 5 に削減"
    echo "2. 解像度を下げる: 1024 → 640 → 512"
    echo "3. gradient_accumulation_steps を増やしてバッチサイズ効果を維持"
    exit $TRAIN_EXIT_CODE
fi