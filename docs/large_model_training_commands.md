# SAM2.1 Hiera-Large モデル実行コマンドガイド

## 概要

SAM2.1 Hiera-Largeモデルでの学習実行コマンドと、Base+からの変更点を説明します。

## 基本実行コマンド比較

### 標準学習

| 用途 | Base+ | Large |
|---|---|---|
| **対話式学習** | `bash scripts/20_train_food_sam2.sh` | `bash scripts/20_train_food_sam2_large.sh` |
| **バックグラウンド学習** | `bash scripts/run_full_training.sh` | `bash scripts/run_full_training_large.sh` |
| **監視付き学習** | `bash scripts/train_with_monitoring.sh --memory-optimized` | `bash scripts/train_with_monitoring.sh --large-model --memory-optimized` |

### 設定ファイル自動生成

| 用途 | Base+ | Large |
|---|---|---|
| **設定ファイル作成** | `bash scripts/create_training_config.sh` | `bash scripts/create_training_config_large.sh` |

### 直接実行コマンド

| 用途 | Base+ | Large |
|---|---|---|
| **標準学習** | `python external/sam2/training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml --use-cluster 0 --num-gpus 1` | `python external/sam2/training/train.py -c configs/sam2.1_training/sam2.1_hiera_l_foodmix_finetune.yaml --use-cluster 0 --num-gpus 1` |
| **メモリ最適化学習** | `python external/sam2/training/train.py -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_optimized.yaml --use-cluster 0 --num-gpus 1` | `python external/sam2/training/train.py -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_optimized.yaml --use-cluster 0 --num-gpus 1` |
| **1エポックテスト** | `python external/sam2/training/train.py -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_test.yaml --use-cluster 0 --num-gpus 1` | `python external/sam2/training/train.py -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_test.yaml --use-cluster 0 --num-gpus 1` |

## 新しいLargeモデル専用オプション

### train_with_monitoring.sh の新オプション

```bash
# Largeモデル使用（自動で最適化設定を選択）
bash scripts/train_with_monitoring.sh --large-model

# Largeモデル + メモリ最適化
bash scripts/train_with_monitoring.sh --large-model --memory-optimized

# Largeモデル + 自動再開
bash scripts/train_with_monitoring.sh --large-model --auto-resume

# カスタム設定
bash scripts/train_with_monitoring.sh --large-model --memory-threshold 85 --gpu-memory-threshold 85
```

### 新しいフラグの説明

| オプション | 説明 | デフォルト値 |
|---|---|---|
| `--large-model` | SAM2.1 Hiera-Largeモデルを使用 | false |
| `--memory-threshold N` | CPUメモリ警告閾値(%) | 90 |
| `--gpu-memory-threshold N` | GPUメモリ警告閾値(%) | 80 |

## GPU メモリ別推奨コマンド

### A100 40GB 以上
```bash
# 標準設定で実行可能
bash scripts/20_train_food_sam2_large.sh
```

### RTX 4090 24GB
```bash
# メモリ最適化推奨
bash scripts/train_with_monitoring.sh --large-model --memory-optimized
```

### RTX 3080/4080 16GB 以下
```bash
# 厳格なメモリ管理
bash scripts/train_with_monitoring.sh --large-model --memory-optimized --gpu-memory-threshold 75
```

## 実行前チェックコマンド

### 環境確認
```bash
# 完全環境チェック
bash scripts/check_large_model_setup.sh

# GPU確認のみ
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

### 設定ファイル確認
```bash
# 作成された設定ファイル一覧
ls -la configs/sam2.1_training/sam2.1_hiera_l_*.yaml
ls -la external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_*.yaml
```

## トラブルシューティング用コマンド

### メモリ不足対策
```bash
# より厳しいメモリ制限
bash scripts/train_with_monitoring.sh --large-model \
    --memory-optimized \
    --gpu-memory-threshold 70 \
    --config sam2.1_training/sam2.1_hiera_l_foodmix_optimized
```

### 学習状況監視
```bash
# 学習ログ監視
tail -f training_logs/training_large_40epochs_*.log

# リソース使用状況リアルタイム監視
watch -n 5 'nvidia-smi; echo ""; free -h'
```

### 学習中断・再開
```bash
# 緊急停止
pkill -f "train.py"

# 自動再開
bash scripts/train_with_monitoring.sh --large-model --auto-resume
```

## パラメータ調整例

### バッチサイズ調整（メモリ不足時）
```bash
# 設定ファイル内で調整
# train_batch_size: 1 → 1 (既に最小)
# gradient_accumulation_steps: 1 → 2 (実効バッチサイズを維持)
```

### オブジェクト数削減（メモリ不足時）
```bash
# max_num_objects を段階的に削減
# 30 → 20 → 15 → 10 → 5
```

### 解像度削減（極端なメモリ不足時）
```bash
# resolution: 1024 → 640
```

## ログファイル出力先

| 実行方法 | ログ出力先 |
|---|---|
| `20_train_food_sam2_large.sh` | `external/sam2/sam2_logs/foodmix_l_finetune/` |
| `run_full_training_large.sh` | `training_logs/training_large_40epochs_*.log` |
| `train_with_monitoring.sh` | `training_monitor.log` + 各設定の標準出力先 |

## チェックポイント保存先

| 設定 | 保存先 |
|---|---|
| **標準学習** | `external/sam2/sam2_logs/foodmix_l_finetune/checkpoints/` |
| **40エポック学習** | `external/sam2/sam2_logs/foodmix_l_40epochs_full/checkpoints/` |
| **メモリ最適化** | `external/sam2/sam2_logs/foodmix_l_optimized/checkpoints/` |
| **テスト学習** | `external/sam2/sam2_logs/foodmix_l_test_1epoch/checkpoints/` |

## 主要な変更点まとめ

### 設定ファイル名変更
- `*_b+_*` → `*_l_*`
- 例: `sam2.1_hiera_b+_foodmix_finetune.yaml` → `sam2.1_hiera_l_foodmix_finetune.yaml`

### ログディレクトリ名変更
- `foodmix_*` → `foodmix_l_*`  
- 例: `foodmix_finetune` → `foodmix_l_finetune`

### チェックポイントファイル変更
- `sam2.1_hiera_base_plus.pt` → `sam2.1_hiera_large.pt`

### モデル構造設定変更
- デフォルトまたは明示定義 → `configs/sam2.1/sam2.1_hiera_l.yaml`

### 実行時間の考慮
- Base+の1.5-2倍の時間が必要
- チェックポイント保存頻度を高く設定（250ステップ）

### メモリ管理の強化
- バッチサイズ1固定
- max_num_objects削減（50→25-30）
- ワーカー数削減（10→2）
- pin_memory無効化

これらの変更により、SAM2.1 Hiera-Largeモデルでの安定した学習実行が可能になります。