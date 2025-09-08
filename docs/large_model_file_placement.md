# SAM2.1 Hiera-Large ファイル配置ガイド

## 概要

SAM2.1 Hiera-Largeモデルへの移行に伴い作成されたファイルの配置先と管理方法を説明します。

## ファイル配置一覧

### 1. チェックポイントファイル

| ファイル | 配置先 | サイズ | 取得方法 |
|---|---|---|---|
| `sam2.1_hiera_large.pt` | `external/sam2/checkpoints/` | ~898MB | 公式ダウンロードスクリプト |

**確認コマンド**:
```bash
ls -la external/sam2/checkpoints/sam2.1_hiera_large.pt
```

### 2. 通常学習用設定ファイル (configs/sam2.1_training/)

プロジェクト直下のメイン設定ディレクトリに配置：

| ファイル | 用途 | Git管理 | 作成状況 |
|---|---|---|---|
| `sam2.1_hiera_l_foodmix_finetune.yaml` | 標準ファインチューニング | ✅ 必要 | ✅ 作成済み |
| `sam2.1_hiera_l_foodmix_simple.yaml` | シンプル学習設定 | ✅ 必要 | ✅ 作成済み |

**配置先**: `/home/soya/sam2_1_food_finetuning/configs/sam2.1_training/`

### 3. メモリ最適化・特殊用途設定ファイル (external/sam2/sam2/configs/sam2.1_training/)

SAM2リポジトリ内の詳細設定ディレクトリに配置：

| ファイル | 用途 | Git管理 | 作成状況 |
|---|---|---|---|
| `sam2.1_hiera_l_foodmix_optimized.yaml` | メモリ最適化版 | ⚠️ サブモジュール | ✅ 作成済み |
| `sam2.1_hiera_l_foodmix_test.yaml` | 1エポックテスト | ⚠️ サブモジュール | ✅ 作成済み |
| `sam2.1_hiera_l_foodmix_40epochs.yaml` | 40エポック フル学習 | ⚠️ サブモジュール | ✅ 作成済み |

**配置先**: `/home/soya/sam2_1_food_finetuning/external/sam2/sam2/configs/sam2.1_training/`

### 4. 実行スクリプト

| ファイル | 配置先 | 実行権限 | Git管理 | 作成状況 |
|---|---|---|---|---|
| `20_train_food_sam2_large.sh` | `scripts/` | ✅ 実行可能 | ✅ 必要 | ✅ 作成済み |

### 5. ドキュメント

| ファイル | 配置先 | Git管理 | 作成状況 |
|---|---|---|---|
| `large_model_optimization_guide.md` | `docs/` | ✅ 必要 | ✅ 作成済み |
| `large_model_config_mapping.md` | `docs/` | ✅ 必要 | ✅ 作成済み |
| `large_model_file_placement.md` | `docs/` | ✅ 必要 | ✅ 作成中 |

## 配置先の理由

### プロジェクト直下 (configs/sam2.1_training/)
- **理由**: Hydra設定システムから直接参照される
- **用途**: 標準的な学習設定
- **管理**: プロジェクトのメインGitリポジトリで管理

### SAM2リポジトリ内 (external/sam2/sam2/configs/sam2.1_training/)
- **理由**: 詳細なトレーナー設定が必要な場合に参照される
- **用途**: メモリ最適化、特殊学習設定
- **管理**: サブモジュール領域（注意が必要）

## Git管理について

### メインリポジトリ管理対象
```bash
# 追加推奨ファイル
git add configs/sam2.1_training/sam2.1_hiera_l_foodmix_finetune.yaml
git add configs/sam2.1_training/sam2.1_hiera_l_foodmix_simple.yaml
git add scripts/20_train_food_sam2_large.sh
git add docs/large_model_*.md
```

### サブモジュール領域の注意事項
`external/sam2/` 内のファイルは以下の特性を持ちます：
- **サブモジュール領域**: 公式SAM2リポジトリのクローン
- **ユーザー追加ファイル**: 学習用設定ファイルを独自に追加
- **管理方法**: 
  - 基本的にはローカルで保持
  - 必要に応じて別途バックアップを推奨
  - サブモジュール更新時に消失する可能性があるため注意

## ファイルアクセス確認

### 存在確認スクリプト
```bash
#!/bin/bash
echo "=== SAM2.1 Large モデル ファイル確認 ==="

echo "1. チェックポイント:"
ls -la external/sam2/checkpoints/sam2.1_hiera_large.pt

echo "2. メイン設定ファイル:"
ls -la configs/sam2.1_training/sam2.1_hiera_l_*.yaml

echo "3. 詳細設定ファイル:"
ls -la external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_*.yaml

echo "4. 実行スクリプト:"
ls -la scripts/20_train_food_sam2_large.sh

echo "5. ドキュメント:"
ls -la docs/large_model_*.md
```

## ファイルの使い分け

### 学習モード別推奨設定

| 学習シナリオ | 推奨設定ファイル | 配置先 |
|---|---|---|
| 標準学習 | `sam2.1_hiera_l_foodmix_finetune.yaml` | configs/ |
| 簡易テスト | `sam2.1_hiera_l_foodmix_simple.yaml` | configs/ |
| メモリ制約環境 | `sam2.1_hiera_l_foodmix_optimized.yaml` | external/sam2/ |
| 1エポックテスト | `sam2.1_hiera_l_foodmix_test.yaml` | external/sam2/ |
| 本格学習 | `sam2.1_hiera_l_foodmix_40epochs.yaml` | external/sam2/ |

## バックアップ推奨事項

サブモジュール領域のファイルは以下の方法でバックアップを推奨：

```bash
# 設定ファイルのバックアップ作成
mkdir -p backup/sam2_configs
cp external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_*.yaml backup/sam2_configs/

# バックアップをメインリポジトリで管理
git add backup/sam2_configs/
git commit -m "backup: SAM2.1 Large model configs"
```

## トラブルシューティング

### ファイルが見つからない場合
1. **チェックポイントファイル**: `cd external/sam2 && bash checkpoints/download_ckpts.sh`
2. **設定ファイル**: このドキュメントの配置先を確認
3. **権限エラー**: `chmod +x scripts/20_train_food_sam2_large.sh`

### サブモジュール更新後の対処
```bash
# 設定ファイルの復元
cp backup/sam2_configs/*.yaml external/sam2/sam2/configs/sam2.1_training/
```

## まとめ

- **メイン設定**: `configs/` でGit管理
- **詳細設定**: `external/sam2/` で個別管理（バックアップ推奨）
- **ドキュメント**: `docs/` で完全管理
- **実行**: `scripts/` で管理
- **チェックポイント**: 公式ダウンロードで取得