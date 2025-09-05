#!/bin/bash
# 20_train_sam2_hydra.sh - SAM2.1の学習実行スクリプト（Hydra対応版）
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

# 設定ファイルの選択
CONFIG_NAME="sam2.1_hiera_b+_foodmix_simple"
CONFIG_PATH="configs/sam2.1_training/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "エラー: 設定ファイルが見つかりません: $CONFIG_PATH"
    exit 1
fi

echo "使用する設定ファイル: $CONFIG_PATH"

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

# SAM2ディレクトリに移動
cd external/sam2

# デフォルト設定ファイルの存在確認
if [ ! -f "configs/sam2.1_training/default.yaml" ]; then
    echo "警告: default.yaml が見つかりません"
    echo "SAM2リポジトリの設定ファイルを確認してください"
fi

# プロジェクトルートの設定ファイルをSAM2ディレクトリにコピー（またはシンボリックリンク）
if [ ! -f "configs/sam2.1_training/${CONFIG_NAME}.yaml" ]; then
    echo "設定ファイルをSAM2ディレクトリにコピー中..."
    cp "../../$CONFIG_PATH" "configs/sam2.1_training/"
fi

# 学習コマンド構築
NUM_GPUS=${NUM_GPUS:-$GPU_COUNT}
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1  # CPU実行の場合
fi

TRAIN_CMD="python training/train.py"
TRAIN_CMD="$TRAIN_CMD -c configs/sam2.1_training/${CONFIG_NAME}.yaml"
TRAIN_CMD="$TRAIN_CMD --num-gpus $NUM_GPUS"
TRAIN_CMD="$TRAIN_CMD --use-cluster 0"

# 実行確認
echo ""
echo "=== 学習コマンド ==="
echo "$TRAIN_CMD"
echo ""
echo "設定:"
echo "  - 設定ファイル: ${CONFIG_NAME}.yaml"
echo "  - GPU数: $NUM_GPUS"
echo "  - 訓練データ: $TRAIN_COUNT"
echo "  - 検証データ: $VAL_COUNT"

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
    echo "チェックポイント: external/sam2/sam2_logs/foodmix_finetune/"
    echo ""
    echo "次のステップ:"
    echo "1. python scripts/21_eval_and_viz.py - モデルの評価と可視化"
else
    echo ""
    echo "=== 学習中にエラーが発生しました ==="
    echo "終了コード: $TRAIN_EXIT_CODE"
    echo ""
    echo "トラブルシューティング:"
    echo "1. 設定ファイルの確認:"
    echo "   - # @package _global_ ディレクティブが先頭にあるか"
    echo "   - パスが正しく設定されているか"
    echo "2. データの確認:"
    echo "   - data/foodmix_sa1b/images と annotations が存在するか"
    echo "   - train.txt と val.txt が正しく生成されているか"
    echo "3. SAM2のインストール確認:"
    echo "   cd external/sam2 && pip install -e ."
    exit $TRAIN_EXIT_CODE
fi