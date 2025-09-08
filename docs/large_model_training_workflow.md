# SAM2.1 Hiera-Large モデル学習ワークフロー

## 完全な学習実行手順

### Phase 1: 事前準備

#### 1.1 環境確認
```bash
# 必須：完全環境チェック
bash scripts/check_large_model_setup.sh

# GPU確認（重要）
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

**チェック項目**：
- ✅ GPU メモリ 20GB以上（推奨）
- ✅ sam2.1_hiera_large.pt 存在
- ✅ Largeモデル設定ファイル 5件存在
- ✅ データセット準備完了

#### 1.2 バックアップ作成（推奨）
```bash
# 設定ファイルのバックアップ
bash scripts/backup_large_model_configs.sh
```

### Phase 2: 学習モード選択

GPU メモリに応じて最適なモードを選択：

#### 2.1 A100 40GB以上: 標準モード
```bash
# 対話式（推奨初回）
bash scripts/20_train_food_sam2_large.sh

# またはバックグラウンド
bash scripts/run_full_training_large.sh
```

#### 2.2 RTX 4090 24GB: メモリ最適化モード
```bash
# 監視付きメモリ最適化（推奨）
bash scripts/train_with_monitoring.sh --large-model --memory-optimized

# または手動設定
python external/sam2/training/train.py \
  -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_optimized.yaml \
  --use-cluster 0 --num-gpus 1
```

#### 2.3 RTX 3080 16GB以下: 厳格最適化モード
```bash
# 厳しいメモリ管理
bash scripts/train_with_monitoring.sh --large-model --memory-optimized \
  --gpu-memory-threshold 75 \
  --memory-threshold 85
```

### Phase 3: 学習実行監視

#### 3.1 初期チェック（最初の10分）
```bash
# GPU使用状況
watch -n 5 nvidia-smi

# 学習ログ確認
tail -f external/sam2/sam2_logs/foodmix_l_*/logs/train.log
```

**確認ポイント**：
- GPU メモリ使用量 < 90%
- OOMエラーなし
- 学習率・損失が正常に変化

#### 3.2 継続監視（バックグラウンド実行時）
```bash
# 監視付きスクリプト使用の場合
tail -f training_monitor.log

# 手動監視の場合
watch -n 30 'ps aux | grep train.py; nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### Phase 4: トラブル対応

#### 4.1 OOM (Out of Memory) エラー
**即座に実行**：
```bash
# 学習停止
pkill -f train.py

# メモリ不足対策版で再開
bash scripts/train_with_monitoring.sh --large-model --memory-optimized \
  --gpu-memory-threshold 70 \
  --auto-resume
```

**設定調整**（必要に応じて）：
- `max_num_objects`: 30 → 20 → 15 → 10
- `resolution`: 1024 → 640
- `gradient_accumulation_steps`: 1 → 2

#### 4.2 学習停滞・異常終了
```bash
# 最新チェックポイントから再開
bash scripts/train_with_monitoring.sh --large-model --auto-resume

# ログの詳細確認
grep -i "error\|warning\|exception" external/sam2/sam2_logs/foodmix_l_*/logs/train.log
```

### Phase 5: 学習完了後の処理

#### 5.1 結果確認
```bash
# 最終チェックポイント確認
ls -la external/sam2/sam2_logs/foodmix_l_*/checkpoints/

# 学習曲線確認
tensorboard --logdir external/sam2/sam2_logs/foodmix_l_*/tensorboard
```

#### 5.2 モデル評価
```bash
# 評価スクリプト実行（Largeモデル用に調整）
python scripts/21_eval_and_viz.py \
  --config external/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint external/sam2/sam2_logs/foodmix_l_*/checkpoints/checkpoint.pt
```

#### 5.3 可視化
```bash
# 予測結果可視化（Largeモデル用に調整）
python scripts/visualize_sam2_predictions.py \
  --model-cfg sam2_hiera_l.yaml \
  --checkpoint external/sam2/sam2_logs/foodmix_l_*/checkpoints/checkpoint.pt
```

## 最適化されたワークフロー例

### クイックスタート（テスト用）
```bash
# 1. 1エポックテスト
python external/sam2/training/train.py \
  -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_test.yaml \
  --use-cluster 0 --num-gpus 1

# 2. 問題なければ本格学習
bash scripts/train_with_monitoring.sh --large-model --memory-optimized
```

### 本格運用（40エポック）
```bash
# 1. 環境確認
bash scripts/check_large_model_setup.sh

# 2. バックアップ
bash scripts/backup_large_model_configs.sh

# 3. 監視付き学習開始
bash scripts/train_with_monitoring.sh --large-model --memory-optimized --auto-resume

# 4. バックグラウンド実行（optional）
nohup bash scripts/train_with_monitoring.sh --large-model --memory-optimized > training_large.log 2>&1 &
```

## 推定学習時間

| GPU | モデル | 40エポック予想時間 | 備考 |
|---|---|---|---|
| A100 40GB | Large | 20-30時間 | 最適条件 |
| RTX 4090 24GB | Large | 30-50時間 | メモリ最適化必要 |
| RTX 3080 16GB | Large | 50-80時間 | 大幅最適化必要 |

**注意**: Base+と比較して1.5-2倍の時間が必要

## チェックポイント管理

### 自動保存設定
- **頻度**: 250ステップごと（Largeモデル用に最適化）
- **保持数**: 最新3個（ディスク容量節約）
- **命名**: `checkpoint_step_*.pt`

### 手動バックアップ
```bash
# 重要チェックポイントのバックアップ
cp external/sam2/sam2_logs/foodmix_l_*/checkpoints/checkpoint.pt \
   backup/large_model_checkpoint_$(date +%Y%m%d).pt
```

## リソース監視ダッシュボード

### 基本監視コマンド
```bash
# GPU + システムリソース
watch -n 5 'nvidia-smi; echo ""; free -h; echo ""; df -h | head -3'

# 学習プロセス詳細
watch -n 10 'ps aux | grep train.py | grep -v grep'
```

### 高度監視（tmux使用）
```bash
# セッション1: 学習実行
tmux new-session -d -s training
tmux send-keys -t training 'bash scripts/train_with_monitoring.sh --large-model --memory-optimized' Enter

# セッション2: リソース監視
tmux new-session -d -s monitoring
tmux send-keys -t monitoring 'watch -n 5 nvidia-smi' Enter

# 接続して確認
tmux attach-session -t training  # または monitoring
```

このワークフローに従うことで、SAM2.1 Hiera-Largeモデルの学習を安全かつ効率的に実行できます。