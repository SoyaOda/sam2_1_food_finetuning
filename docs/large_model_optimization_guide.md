# SAM2.1 Hiera-Large モデル最適化ガイド

## 概要
SAM2.1 Hiera-Largeモデルは Base+ より大きなパラメータ数を持つため、メモリ使用量と計算コストが増加します。このガイドでは、効率的な学習のための最適化設定を説明します。

## 設定ファイル

### 通常学習用
- **ファイル**: `configs/sam2.1_training/sam2.1_hiera_l_foodmix_finetune.yaml`
- **用途**: 標準的な学習環境（32GB以上のGPU推奨）

### メモリ最適化版
- **ファイル**: `external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_optimized.yaml`
- **用途**: メモリ制限のある環境（16GB GPU対応）

## 主要な最適化パラメータ

### メモリ使用量削減
- `train_batch_size: 1` - バッチサイズを最小に
- `max_num_objects: 25` - 同時処理オブジェクト数を削減（Base+: 50 → Large: 25）
- `num_train_workers: 2` - データローダーワーカー数を削減
- `pin_memory: false` - メモリリーク防止

### 学習効率最適化
- `amp_dtype: bfloat16` - 混合精度でメモリ効率向上
- `base_lr: 1.0e-4` - SAM2.1推奨範囲内での安定学習率
- `save_freq: 250` - 適切な間隔でのチェックポイント保存
- `log_freq: 100` - ログオーバーヘッド削減

## GPU別推奨設定

### A100 40GB
- バッチサイズ: 1-2
- max_num_objects: 25-30
- 混合精度: bfloat16

### RTX 4090 24GB
- バッチサイズ: 1
- max_num_objects: 15-20
- 混合精度: bfloat16

### RTX 3080/4080 16GB以下
- バッチサイズ: 1
- max_num_objects: 10-15
- 混合精度: fp16またはbfloat16
- 解像度削減も検討: 1024 → 640

## メモリ不足時の対応

### 段階的対応
1. max_num_objects を 25 → 15 → 10 → 5 に削減
2. 解像度を 1024 → 640 → 512 に削減
3. gradient_accumulation_steps を増やしてバッチサイズ効果を維持

### 極端なメモリ制限時
- max_num_objects: 3-5
- 解像度: 512×512
- ワーカー数: 1
- gradient_accumulation_steps: 4-8

## 計算時間の考慮

Large モデルは Base+ より約1.5-2倍の計算時間がかかります：

### 推定学習時間
| GPU | 40エポック予想時間 | 備考 |
|---|---|---|
| A100 40GB | 20-30時間 | 最適条件 |
| RTX 4090 24GB | 30-50時間 | メモリ最適化必要 |
| RTX 3080 16GB | 50-80時間 | 大幅最適化必要 |

### 時間短縮の工夫
- チェックポイント保存間隔を短縮（250ステップ推奨）
- 評価頻度を下げる（オーバーヘッド削減）
- 初回は短いエポック数でテスト実行推奨
- バックグラウンド実行で継続学習

### メモリ使用量の詳細影響

#### 学習時の段階別メモリ消費
1. **モデル読み込み**: ~2-3GB（Base+の約4倍）
2. **Forward Pass**: ~8-12GB（解像度・オブジェクト数依存）
3. **Backward Pass**: ~12-18GB（勾配計算）
4. **Optimizer State**: ~6-8GB（AdamW使用時）
5. **バッファ・その他**: ~2-4GB

**合計推定**: 30-45GB（ピーク時）

#### 緊急時メモリ削減設定
```yaml
# 極限メモリ削減設定
max_num_objects: 5      # 通常30から5へ
resolution: 512         # 1024から512へ
train_batch_size: 1     # 固定
num_train_workers: 1    # 2から1へ
pin_memory: false       # メモリリーク防止
gradient_accumulation_steps: 4  # 実効バッチサイズ維持
```

## 実行例

```bash
# 通常学習
bash scripts/20_train_food_sam2.sh configs/sam2.1_training/sam2.1_hiera_l_foodmix_finetune.yaml

# メモリ最適化学習
bash scripts/train_with_monitoring.sh --config sam2.1_training/sam2.1_hiera_l_foodmix_optimized --memory-optimized
```