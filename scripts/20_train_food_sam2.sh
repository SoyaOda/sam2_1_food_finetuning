#!/bin/bash
# 20_train_food_sam2.sh - SAM2.1の学習実行スクリプト
set -eux

echo "SAM2.1 Food Finetuning 学習を開始します..."

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

# 設定ファイルの存在確認
CONFIG_FILE="configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "警告: $CONFIG_FILE が見つかりません"
    echo "設定ファイルを作成します..."
    
    # ディレクトリ作成
    mkdir -p configs/sam2.1_training
    
    # 基本的な設定ファイルを作成（後で詳細版に置き換え）
    echo "設定ファイルの作成はこの後のステップで行います"
    echo "一旦、SAM2のサンプル設定を確認します..."
fi

# GPU確認
echo ""
echo "=== GPU情報 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "検出されたGPU数: $GPU_COUNT"
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

# メモリ推定（簡易）
if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -lt 16000 ]; then
        echo ""
        echo "警告: GPUメモリが16GB未満です（${GPU_MEM}MB）"
        echo "バッチサイズを1に設定することを推奨します"
        BATCH_SIZE=1
    elif [ "$GPU_MEM" -lt 24000 ]; then
        echo "GPUメモリ: ${GPU_MEM}MB - バッチサイズ1-2を推奨"
        BATCH_SIZE=1
    else
        echo "GPUメモリ: ${GPU_MEM}MB - バッチサイズ2-4を推奨"
        BATCH_SIZE=2
    fi
else
    BATCH_SIZE=1
fi

# 学習パラメータ設定
NUM_GPUS=${NUM_GPUS:-$GPU_COUNT}
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1  # CPU実行の場合
fi

# 実行モード選択
echo ""
echo "=== 学習モード選択 ==="
echo "1. 設定ファイルを作成して学習開始"
echo "2. 既存の設定ファイルで学習開始"
echo "3. 設定ファイルの作成のみ"
echo "4. ドライラン（設定確認のみ）"

read -p "選択してください (1-4): " MODE

case $MODE in
    1)
        # 設定ファイル作成して学習
        echo "設定ファイルを作成中..."
        bash scripts/create_training_config.sh
        ;;
    2)
        # 既存設定で学習
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "エラー: 設定ファイルが見つかりません"
            exit 1
        fi
        ;;
    3)
        # 設定作成のみ
        bash scripts/create_training_config.sh
        echo "設定ファイルを作成しました: $CONFIG_FILE"
        echo "学習を開始するには再度このスクリプトを実行してください"
        exit 0
        ;;
    4)
        # ドライラン
        echo "ドライランモード..."
        DRY_RUN=true
        ;;
    *)
        echo "無効な選択です"
        exit 1
        ;;
esac

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
echo "=== 学習コマンド ==="
echo "$TRAIN_CMD"
echo ""
echo "設定:"
echo "  - GPU数: $NUM_GPUS"
echo "  - バッチサイズ（推奨）: $BATCH_SIZE"
echo "  - 訓練データ: $TRAIN_COUNT"
echo "  - 検証データ: $VAL_COUNT"

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo ""
    echo "ドライラン完了。実際の学習は実行されませんでした。"
    exit 0
fi

echo ""
read -p "学習を開始しますか？ (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "学習を中止しました"
    exit 0
fi

# 学習実行
echo ""
echo "学習を開始します..."
echo "ログは external/sam2/sam2_logs/ に保存されます"
echo "Ctrl+C で中断できます（チェックポイントから再開可能）"
echo ""

# 学習実行（エラーハンドリング付き）
set +e
$TRAIN_CMD
TRAIN_EXIT_CODE=$?
set -e

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== 学習が正常に完了しました！ ==="
    echo "チェックポイント: external/sam2/sam2_logs/foodmix_bplus_40ep/checkpoints/"
    echo ""
    echo "次のステップ:"
    echo "1. python scripts/21_eval_and_viz.py - モデルの評価と可視化"
else
    echo ""
    echo "=== 学習中にエラーが発生しました ==="
    echo "終了コード: $TRAIN_EXIT_CODE"
    echo ""
    echo "トラブルシューティング:"
    echo "1. GPUメモリ不足の場合: 設定ファイルのbatch_sizeを小さくしてください"
    echo "2. データエラーの場合: 前処理スクリプトを再実行してください"
    echo "3. 依存関係エラーの場合: bash scripts/00_setup_env.sh を再実行してください"
    exit $TRAIN_EXIT_CODE
fi