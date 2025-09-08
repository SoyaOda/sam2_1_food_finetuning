# SAM2.1 Base+ → Large モデル設定ファイル対応表

## 概要

このドキュメントでは、既存のSAM2.1 Hiera Base+用設定ファイルに対応するLargeモデル用設定ファイルの対応関係を示します。

## 設定ファイル対応表

### プロジェクト直下の設定ファイル (`configs/sam2.1_training/`)

| Base+ 設定ファイル | Large 設定ファイル | 用途 | 状態 |
|---|---|---|---|
| `sam2.1_hiera_b+_foodmix_finetune.yaml` | `sam2.1_hiera_l_foodmix_finetune.yaml` | 通常学習用（フル設定） | ✅ 作成済み |
| `sam2.1_hiera_b+_foodmix_simple.yaml` | `sam2.1_hiera_l_foodmix_simple.yaml` | シンプル設定 | ✅ 作成済み |

### SAM2リポジトリ内の設定ファイル (`external/sam2/sam2/configs/sam2.1_training/`)

| Base+ 設定ファイル | Large 設定ファイル | 用途 | 状態 |
|---|---|---|---|
| `sam2.1_hiera_b+_foodmix_optimized.yaml` | `sam2.1_hiera_l_foodmix_optimized.yaml` | メモリ最適化版 | ✅ 作成済み |
| `sam2.1_hiera_b+_foodmix_test.yaml` | `sam2.1_hiera_l_foodmix_test.yaml` | テスト用（1エポック） | ✅ 作成済み |
| `sam2.1_hiera_b+_foodmix_40epochs.yaml` | `sam2.1_hiera_l_foodmix_40epochs.yaml` | 40エポック フル学習 | ✅ 作成済み |

### その他の設定ファイル（必要に応じて作成可能）

| Base+ 設定ファイル | Large 設定ファイル | 用途 | 状態 |
|---|---|---|---|
| `sam2.1_hiera_b+_simple_subset.yaml` | `sam2.1_hiera_l_simple_subset.yaml` | サブセットテスト | 📝 未作成 |
| `sam2.1_hiera_b+_subset_test.yaml` | `sam2.1_hiera_l_subset_test.yaml` | サブセットテスト | 📝 未作成 |
| `sam2.1_hiera_b+_test_finetune.yaml` | `sam2.1_hiera_l_test_finetune.yaml` | テストファインチューン | 📝 未作成 |

## 主な変更点

### 1. チェックポイントファイル
- **Base+**: `sam2.1_hiera_base_plus.pt`
- **Large**: `sam2.1_hiera_large.pt`

### 2. モデル設定参照
- **Base+**: 設定ファイル内でモデル構造を明示的に定義またはデフォルト使用
- **Large**: `model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml` で公式設定を参照

### 3. メモリ最適化パラメータ

| パラメータ | Base+ | Large | 理由 |
|---|---|---|---|
| `train_batch_size` | 1-2 | 1 | メモリ使用量増加 |
| `max_num_objects` | 50 | 25-30 | オブジェクト処理のメモリ削減 |
| `num_train_workers` | 10 | 2 | メモリ効率化 |
| `pin_memory` | true | false | メモリリーク防止 |
| `amp_dtype` | float16 | bfloat16 | より安定した混合精度 |
| `base_lr` | 5.0e-6 | 1.0e-4 | SAM2.1推奨範囲内 |
| `save_freq` | 200-5000 | 250 | 時間短縮のための頻繁保存 |
| `log_freq` | 10-50 | 50-100 | ログオーバーヘッド削減 |

### 4. ディレクトリ命名規則

- **Base+**: `foodmix_*` (例: `foodmix_finetune`, `foodmix_test_1epoch`)
- **Large**: `foodmix_l_*` (例: `foodmix_l_finetune`, `foodmix_l_test_1epoch`)

## 使用方法

### 通常学習
```bash
# Base+
bash scripts/20_train_food_sam2.sh

# Large
bash scripts/20_train_food_sam2_large.sh
```

### メモリ最適化学習
```bash
# Base+
bash scripts/train_with_monitoring.sh --config sam2.1_training/sam2.1_hiera_b+_foodmix_optimized --memory-optimized

# Large
bash scripts/train_with_monitoring.sh --config sam2.1_training/sam2.1_hiera_l_foodmix_optimized --memory-optimized
```

### テスト実行
```bash
# Large 1エポックテスト
python external/sam2/training/train.py -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_test.yaml --use-cluster 0 --num-gpus 1
```

## 新しい設定ファイルを作成する場合

1. 既存のBase+設定ファイルをコピー
2. 以下の変更を適用：
   - `checkpoint` パスを Large 用に変更
   - `model_cfg` を追加（必要に応じて）
   - メモリ最適化パラメータを調整
   - ログディレクトリ名を `*_l_*` に変更
   - コメントでLarge版であることを明記

## 注意事項

- Largeモデルは計算時間が1.5-2倍かかります
- メモリ使用量が大幅に増加するため、適切なGPU環境が必要です
- 初回実行時は短いエポック数でテストすることを推奨します
- メモリ不足が発生した場合は、メモリ最適化版設定を使用してください