# SAM2.1 Food Segmentation Training System

食品セグメンテーション用SAM2.1トレーニングシステムの完全実装ガイド

## 🎯 プロジェクト概要

本プロジェクトは、SAM2.1（Segment Anything Model 2.1）を食品画像データセットでファインチューニングするための包括的なトレーニングシステムです。Hiera-Base+からHiera-Largeモデルまでの統合的な実装を提供しています。

### 主要機能
- ✅ SAM2.1 Hiera-Base+とHiera-Largeモデル対応
- ✅ FoodSeg103+UECFoodPIX統合データセット
- ✅ メモリ最適化トレーニングパイプライン
- ✅ 包括的なパフォーマンス監視システム
- ✅ 自動モデル移行・設定システム
- ✅ 推論・評価・可視化ツール

## 📁 プロジェクト構造

```
sam2_1_food_finetuning/
├── README_COMPLETE.md              # 本ファイル（完全ガイド）
├── CLAUDE.md                       # Claude Code指示書
├── external/sam2/                  # SAM2.1 サブモジュール
├── configs/sam2.1_training/        # トレーニング設定ファイル
│   ├── sam2.1_hiera_b+_foodmix_finetune.yaml    # Base+モデル本格学習
│   ├── sam2.1_hiera_b+_foodmix_simple.yaml      # Base+モデルテスト学習
│   ├── sam2.1_hiera_l_foodmix_finetune.yaml     # Largeモデル本格学習
│   └── sam2.1_hiera_l_foodmix_simple.yaml       # Largeモデルテスト学習
├── training/                       # カスタムトレーニングコード
│   └── memory_optimized_trainer.py # メモリ最適化トレーナー
├── scripts/                        # 実行スクリプト集
│   ├── 00_setup_env.sh            # 環境セットアップ
│   ├── 01_download_foodseg103_hf.py # データダウンロード
│   ├── 02_download_uecfoodpix.sh  # UECFoodPIX取得
│   ├── 10_prepare_foodseg103.py   # データ前処理
│   ├── 11_prepare_uecfoodpix.py   # UECFoodPIX前処理
│   ├── 12_merge_to_sa1b.py        # SA1B形式統合
│   ├── 20_train_food_sam2.sh      # Base+学習スクリプト
│   ├── 20_train_food_sam2_large.sh # Large学習スクリプト
│   ├── 21_eval_and_viz.py         # 評価・可視化
│   ├── visualize_sam2_predictions.py      # Base+推論
│   ├── visualize_sam2_predictions_large.py # Large推論
│   └── train_with_monitoring.sh   # 監視付き学習
├── docs/                           # ドキュメント
│   ├── large_model_optimization_guide.md      # Largeモデル最適化
│   └── large_model_performance_monitoring.md  # パフォーマンス監視
└── md_files/                       # 仕様書
    ├── spec1.md                    # 基本実装仕様
    └── large_model_shift_spec.md   # Largeモデル移行仕様
```

## ⚙️ システム要件

### 必須環境
- **GPU**: NVIDIA GPU (24GB+ VRAM推奨、Largeモデル使用時)
- **Python**: 3.9+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+

### 推奨環境
- **Base+モデル**: RTX 3090/4090 (24GB)
- **Largeモデル**: RTX 6000 Ada/A6000 (48GB)またはA100 (40GB/80GB)

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
# リポジトリクローン
git clone <repository-url>
cd sam2_1_food_finetuning

# 基本環境セットアップ
bash scripts/00_setup_env.sh

# SAM2.1サブモジュール初期化
cd external/sam2
git submodule update --init --recursive
cd ../..

# 依存関係インストール
pip install -e external/sam2
```

### 2. データ準備
```bash
# FoodSeg103ダウンロード（HuggingFace経由）
python scripts/01_download_foodseg103_hf.py

# UECFoodPIXダウンロード（オプション）
bash scripts/02_download_uecfoodpix.sh

# データ前処理とSA1B形式変換
python scripts/10_prepare_foodseg103.py
python scripts/11_prepare_uecfoodpix.py
python scripts/12_merge_to_sa1b.py
```

### 3. テスト学習（動作確認）
```bash
# Base+モデルのテスト学習（数サンプルで動作確認）
export HYDRA_FULL_ERROR=1
cd external/sam2
python training/train.py \
    -c sam2.1_training/sam2.1_hiera_b+_foodmix_simple \
    --use-cluster 0 --num-gpus 1

# Largeモデルのテスト学習
python training/train.py \
    -c sam2.1_training/sam2.1_hiera_l_foodmix_simple \
    --use-cluster 0 --num-gpus 1
```

## 🏋️ 本格学習コマンド

### Base+モデル（24GB GPU推奨）
```bash
# 標準学習
cd external/sam2
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:/home/soya/sam2_1_food_finetuning/external/sam2"

python training/train.py \
    -c sam2.1_training/sam2.1_hiera_b+_foodmix_finetune \
    --use-cluster 0 --num-gpus 1

# メモリ最適化学習（低メモリ環境用）
bash ../../scripts/train_with_monitoring.sh sam2.1_hiera_b+_foodmix_finetune
```

### Largeモデル（48GB+ GPU推奨）
```bash
# 標準学習
cd external/sam2
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:/home/soya/sam2_1_food_finetuning/external/sam2"

python training/train.py \
    -c sam2.1_training/sam2.1_hiera_l_foodmix_finetune \
    --use-cluster 0 --num-gpus 1

# 極限メモリ最適化学習
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
python training/train.py \
    -c sam2.1_training/sam2.1_hiera_l_foodmix_finetune \
    --use-cluster 0 --num-gpus 1
```

### 監視付き学習（推奨）
```bash
# GPU監視付き学習実行
bash scripts/train_with_monitoring.sh sam2.1_hiera_l_foodmix_finetune

# 別ターミナルでリアルタイム監視
watch -n 2 nvidia-smi
```

## 🔍 推論・評価コマンド

### Base+モデル推論
```bash
# 基本推論実行
python scripts/visualize_sam2_predictions.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --image_dir data/food_dataset/val/images \
    --output_dir outputs/base_plus_predictions

# 評価付き推論
python scripts/21_eval_and_viz.py \
    --model base_plus \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --test_data data/food_dataset/val
```

### Largeモデル推論
```bash
# 基本推論実行
python scripts/visualize_sam2_predictions_large.py \
    --checkpoint checkpoints/sam2.1_hiera_large.pt \
    --config_path sam2.1_hiera_l.yaml \
    --image_dir data/food_dataset/val/images \
    --output_dir outputs/large_model_predictions

# メモリ最適化推論
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
python scripts/visualize_sam2_predictions_large.py \
    --checkpoint checkpoints/sam2.1_hiera_large.pt \
    --image_dir data/food_dataset/val/images \
    --output_dir outputs/large_model_predictions \
    --batch_size 1 \
    --low_memory_mode
```

### パフォーマンス比較
```bash
# Base+ vs Large モデル比較実行
echo "=== Base+ Model Performance ===" && \
time python scripts/visualize_sam2_predictions.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --image_dir data/food_dataset/val/images \
    --output_dir outputs/base_plus_predictions && \
echo "=== Large Model Performance ===" && \
time python scripts/visualize_sam2_predictions_large.py \
    --checkpoint checkpoints/sam2.1_hiera_large.pt \
    --image_dir data/food_dataset/val/images \
    --output_dir outputs/large_model_predictions
```

## 📊 パフォーマンス監視

### GPUメモリ監視
```bash
# リアルタイムGPU監視
python -c "
import torch
import time
import datetime
while True:
    if torch.cuda.is_available():
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        mem_used = torch.cuda.memory_allocated()/1024**3
        mem_max = torch.cuda.max_memory_allocated()/1024**3
        print(f'[{timestamp}] GPU Memory: {mem_used:.2f}GB / {mem_max:.2f}GB')
    time.sleep(5)
"

# nvidia-smi監視
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,nounits,noheader
```

### 学習監視付き実行
```bash
# バックグラウンドでGPU監視＋メイン学習処理
(
    # GPU監視（バックグラウンド）
    python -c "
import torch, time, datetime
while True:
    if torch.cuda.is_available():
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        mem_used = torch.cuda.memory_allocated()/1024**3
        print(f'[{timestamp}] GPU Memory: {mem_used:.2f}GB')
    time.sleep(10)
" &
    
    # メイン学習処理
    cd external/sam2
    python training/train.py -c sam2.1_training/sam2.1_hiera_l_foodmix_finetune --use-cluster 0 --num-gpus 1
    
    # 監視プロセス終了
    kill %1
)
```

## 🧪 デバッグ・テストコマンド

### 設定ファイル検証
```bash
# Hydra設定の検証
cd external/sam2
python -c "
from hydra import compose, initialize_config_module
initialize_config_module('sam2', version_base='1.2')
cfg = compose(config_name='sam2.1_training/sam2.1_hiera_l_foodmix_finetune')
print('設定検証成功!')
print(f'データセット: {cfg.dataset.img_folder}')
print(f'バッチサイズ: {cfg.scratch.train_batch_size}')
print(f'学習率: {cfg.scratch.base_lr}')
"
```

### データローダーテスト
```bash
# データセット読み込みテスト
python scripts/test_train_minimal.py

# データ構造確認
python scripts/check_data_structure.py

# SA1B形式検証
python scripts/verify_sa1b_data.py
```

### モデル読み込みテスト
```bash
# Base+モデル読み込み確認
python -c "
from sam2.build_sam import build_sam2
model = build_sam2('configs/sam2_hiera_b+.yaml', 'checkpoints/sam2.1_hiera_base_plus.pt')
print('Base+ model loaded successfully!')
"

# Largeモデル読み込み確認
python -c "
from sam2.build_sam import build_sam2
model = build_sam2('configs/sam2.1_hiera_l.yaml', 'checkpoints/sam2.1_hiera_large.pt')
print('Large model loaded successfully!')
"
```

## 🔧 トラブルシューティング

### メモリ不足エラー
```bash
# メモリ使用量削減設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# バッチサイズ削減（設定ファイル修正）
# train_batch_size: 1 または 2 に設定

# gradient_accumulationの活用
# effective_batch_size = train_batch_size * gradient_accumulation_steps
```

### GPU認識エラー
```bash
# CUDA環境確認
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# nvidia-smi確認
nvidia-smi
```

### Hydra設定エラー
```bash
# エラー詳細表示設定
export HYDRA_FULL_ERROR=1

# 設定ファイルパス確認
ls -la configs/sam2.1_training/
ls -la external/sam2/sam2/configs/sam2.1_training/
```

## 📈 期待される結果

### 学習進度の目安
- **Base+モデル**: 
  - 初期loss: ~0.8-1.0
  - 収束loss: ~0.3-0.5
  - 学習時間: 24時間程度（1エポック約2-3時間）

- **Largeモデル**:
  - 初期loss: ~0.7-0.9  
  - 収束loss: ~0.2-0.4
  - 学習時間: 36-48時間程度（1エポック約4-5時間）

### パフォーマンス指標
- **Base+**: IoU 0.75-0.85, mAP 0.70-0.80
- **Large**: IoU 0.80-0.90, mAP 0.75-0.85

## 🎯 実装済み機能詳細

### 1. モデル設定システム
- ✅ Hiera-Base+ ⟷ Hiera-Large自動移行
- ✅ 設定ファイル統合管理
- ✅ チェックポイント自動対応

### 2. メモリ最適化システム
- ✅ Gradient checkpointing
- ✅ Mixed precision training (FP16/BF16)
- ✅ Batch size動的調整
- ✅ メモリ監視・アラート

### 3. データ処理パイプライン
- ✅ FoodSeg103 + UECFoodPIX統合
- ✅ SA1B形式自動変換
- ✅ データ前処理・検証

### 4. 学習システム
- ✅ Hydra設定管理
- ✅ 分散学習対応
- ✅ チェックポイント・レジューム
- ✅ 学習監視システム

### 5. 評価・可視化システム
- ✅ 推論スクリプト（Base+/Large対応）
- ✅ IoU/mAP評価
- ✅ 可視化ツール
- ✅ パフォーマンス比較

## 📚 参考資料

### 重要ファイル
- `md_files/spec1.md`: 基本実装仕様
- `md_files/large_model_shift_spec.md`: Largeモデル移行仕様  
- `docs/large_model_optimization_guide.md`: 最適化ガイド
- `docs/large_model_performance_monitoring.md`: 監視ガイド

### 公式リンク
- [SAM2.1公式リポジトリ](https://github.com/facebookresearch/sam2)
- [FoodSeg103データセット](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103)
- [UECFoodPIXデータセット](http://foodcam.mobi/dataset256.html)

---

**作成日**: 2025-01-08  
**最終更新**: SAM2.1 Largeモデル実装完了
**ステータス**: 本格運用可能

このREADMEは、SAM2.1食品セグメンテーションシステムの完全実装ガイドです。質問やサポートが必要な場合は、プロジェクト管理者にお問い合わせください。