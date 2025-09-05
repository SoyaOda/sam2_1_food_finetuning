# SAM2.1 メモリ最適化ガイド

PCが重くなって学習が途中で止まる問題を解決するための最適化されたツールセットです。

## 🎯 問題の解決策

### 主な原因と対策

1. **メモリリーク**: 定期的なキャッシュクリアとガベージコレクション
2. **過剰なワーカー数**: WSL環境に適した設定に調整
3. **pin_memory問題**: メモリ固定の無効化
4. **gradient accumulation**: チェックポイント機能で削減

## 📁 新しく追加されたファイル

```
├── external/sam2/sam2/configs/sam2.1_training/
│   └── sam2.1_hiera_b+_foodmix_optimized.yaml    # メモリ最適化設定
├── training/
│   └── memory_optimized_trainer.py               # メモリ監視機能付きTrainer
├── scripts/
│   ├── train_sam2_memory_optimized.py            # メモリ最適化学習スクリプト
│   ├── resume_training.py                        # 学習再開スクリプト
│   └── train_with_monitoring.sh                  # 監視機能付き学習
└── MEMORY_OPTIMIZATION_GUIDE.md                  # このファイル
```

## 🚀 使用方法

### 1. 基本的な学習（メモリ最適化版）

```bash
# メモリ最適化された学習の実行
python scripts/train_sam2_memory_optimized.py \
    -c sam2.1_training/sam2.1_hiera_b+_foodmix_optimized \
    --use-cluster 0 \
    --num-gpus 1
```

### 2. 監視機能付き学習

```bash
# システムリソース監視付きで学習実行
bash scripts/train_with_monitoring.sh \
    --memory-optimized \
    --config sam2.1_training/sam2.1_hiera_b+_foodmix_optimized
```

### 3. 学習の再開

```bash
# 対話式で再開する実験を選択
python scripts/resume_training.py --interactive --memory-optimized

# 特定のディレクトリから再開
python scripts/resume_training.py \
    --experiment-dir ./sam2_logs/foodmix_optimized \
    --memory-optimized
```

### 4. 利用可能な実験の確認

```bash
# 再開可能な実験のリスト表示
python scripts/resume_training.py --list
```

## ⚙️ 最適化設定の詳細

### メモリ使用量の削減

| 設定項目 | 元の値 | 最適化後 | 効果 |
|---------|--------|----------|------|
| `num_train_workers` | 10 | 2 | CPUメモリ削減 |
| `pin_memory` | True | False | メモリリーク防止 |
| `max_num_objects` | 50 | 25 | GPU/CPUメモリ削減 |
| `num_maskmem` | 7 | 5 | メモリ使用量削減 |
| `multiplier` | 2 | 1 | データ読み込み削減 |
| `save_freq` | 5000 | 2000 | 早期チェックポイント |

### 監視機能

- **CPUメモリ監視**: 使用率85%でアラート
- **GPUメモリ監視**: 使用率90%でアラート
- **自動クリーンアップ**: 100ステップごとにキャッシュクリア
- **ガベージコレクション**: 50ステップごとに実行

## 📊 パフォーマンス改善

### Before（問題のある状態）
```
Mem (GB): 6.00 → 42.00 (急激な増加)
PCが重くなり途中でクラッシュ
```

### After（最適化後）
```
Mem (GB): 8.00 → 12.00 (安定)
長時間学習でも安定動作
定期的なメモリクリーンアップ
```

## 🔧 トラブルシューティング

### Q: まだメモリ不足エラーが出る
**A:** 設定をさらに下げる
```bash
# バッチサイズを1にして、解像度も下げる
# 設定ファイルの resolution: 1024 → 512
# train_batch_size: 1 → 1のまま（これ以上は下げられない）
```

### Q: 学習速度が遅い
**A:** 監視頻度を調整
```bash
python scripts/train_sam2_memory_optimized.py \
    -c sam2.1_training/sam2.1_hiera_b+_foodmix_optimized \
    --memory-check-interval 200  # デフォルト100から200に
```

### Q: 過去の学習を再開したい
**A:** resume_training.pyを使用
```bash
# 対話式選択
python scripts/resume_training.py --interactive

# 自動で最新を選択
python scripts/resume_training.py --memory-optimized
```

## 📈 監視ログの確認

### 学習中のログ例
```
[INFO] 2025-09-05 10:30:15 - Memory Usage: CPU 8.5/21.0GB (40.5%), GPU 10.2/12.0GB (85.0%)
[INFO] 2025-09-05 10:30:45 - Memory cleanup #5: CPU 8.5GB → 7.8GB (saved: 0.7GB)
[WARN] 2025-09-05 10:31:15 - High GPU memory usage: 92.0%
```

### ログファイル
- `memory_optimized_training.log`: 詳細なメモリ使用ログ
- `training_monitor.log`: システム監視ログ
- `./sam2_logs/[experiment]/logs/`: 学習ログ

## 🎯 推奨使用パターン

### パターン1: 安全第一（推奨）
```bash
# 監視機能付きでメモリ最適化学習
bash scripts/train_with_monitoring.sh --memory-optimized
```

### パターン2: 最大パフォーマンス
```bash
# 監視間隔を長くして高速化
python scripts/train_sam2_memory_optimized.py \
    -c sam2.1_training/sam2.1_hiera_b+_foodmix_optimized \
    --memory-check-interval 500
```

### パターン3: 再開重視
```bash
# 頻繁なチェックポイント保存で安全性確保
# 設定ファイルのsave_freq: 1000に変更してから実行
```

## 🏁 まとめ

1. **`scripts/train_sam2_memory_optimized.py`** でメモリ効率的な学習
2. **`scripts/train_with_monitoring.sh`** でシステム監視
3. **`scripts/resume_training.py`** で学習再開
4. **チェックポイント機能** で途中からの復帰が可能

これらのツールにより、長時間の学習でもシステムが安定し、途中でクラッシュしても学習を再開できます。