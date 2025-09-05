#!/bin/bash

# SAM2.1の学習スクリプトをdry-runモードでテスト
# Hydra設定の読み込みとデータローディングの確認

set -e

echo "============================================"
echo "SAM2.1 学習スクリプト動作テスト (Dry-run)"
echo "============================================"

cd external/sam2

# 設定ファイルの存在確認
CONFIG_FILE="sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "エラー: 設定ファイルが見つかりません: $CONFIG_FILE"
    exit 1
fi

echo "設定ファイル: $CONFIG_FILE"

# チェックポイントの存在確認
CHECKPOINT="checkpoints/sam2.1_hiera_base_plus.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "エラー: チェックポイントが見つかりません: $CHECKPOINT"
    echo "ダウンロードしてください: cd checkpoints && ./download_ckpts.sh"
    exit 1
fi

echo "チェックポイント: $CHECKPOINT"

# データディレクトリの確認
DATA_DIR="../../data/foodmix_sa1b"
if [ ! -d "$DATA_DIR/images" ] || [ ! -d "$DATA_DIR/annotations" ]; then
    echo "エラー: データディレクトリが見つかりません"
    echo "  画像: $DATA_DIR/images"
    echo "  アノテーション: $DATA_DIR/annotations"
    exit 1
fi

echo "データディレクトリ: $DATA_DIR"

# Hydra設定のデバッグ出力を有効にして実行（1ステップのみ）
echo ""
echo "学習スクリプトを起動します（1ステップのみのテスト実行）..."
echo ""

# 環境変数設定
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# dry-run用の一時設定を作成
TEMP_CONFIG="/tmp/sam2_test_config.yaml"
cat > $TEMP_CONFIG << EOF
# @package _global_

defaults:
  - sam2.1_training/sam2.1_hiera_b+_foodmix_finetune

# テスト用の設定オーバーライド
trainer:
  max_epochs: 1  # 1エポックのみ

launcher:
  experiment_log_dir: /tmp/sam2_test_logs

# データ設定は元の設定を使用
EOF

# Pythonスクリプトでテスト実行
python3 -c "
import sys
import os
sys.path.insert(0, '.')

# Hydraの初期化
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

initialize_config_module('sam2', version_base='1.2')

# 設定の読み込みとデバッグ出力
try:
    cfg = compose(config_name='sam2.1_training/sam2.1_hiera_b+_foodmix_finetune')
    print('=== Hydra設定の読み込み成功 ===')
    print()
    print('主要な設定項目:')
    print(f'  学習データ: {cfg.dataset.img_folder}')
    print(f'  アノテーション: {cfg.dataset.gt_folder}')
    print(f'  学習リスト: {cfg.dataset.file_list_txt}')
    print(f'  バッチサイズ: {cfg.scratch.train_batch_size}')
    print(f'  エポック数: {cfg.scratch.num_epochs}')
    print(f'  学習率: {cfg.scratch.base_lr}')
    print()
    
    # データローダーのテスト
    from training.dataset.vos_raw_dataset import SA1BRawDataset
    print('=== データセットの読み込みテスト ===')
    
    dataset = SA1BRawDataset(
        img_folder=cfg.dataset.img_folder,
        gt_folder=cfg.dataset.gt_folder,
        file_list_txt=cfg.dataset.file_list_txt
    )
    
    print(f'データセットサイズ: {len(dataset)}')
    
    # サンプルデータの取得
    sample = dataset[0]
    print(f'サンプルデータの形状:')
    if 'images' in sample:
        print(f'  画像: {sample[\"images\"].shape if hasattr(sample[\"images\"], \"shape\") else type(sample[\"images\"])}')
    if 'masks' in sample:
        print(f'  マスク: {sample[\"masks\"].shape if hasattr(sample[\"masks\"], \"shape\") else type(sample[\"masks\"])}')
    
    print()
    print('✅ すべてのテストが成功しました！')
    print('実際の学習を開始する場合は以下のコマンドを実行してください:')
    print()
    print('cd external/sam2')
    print('python training/train.py -c sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml --use-cluster 0 --num-gpus 1')
    
except Exception as e:
    print(f'エラー: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "============================================"
echo "テスト完了"
echo "============================================"