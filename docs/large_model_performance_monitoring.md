# SAM2.1 Hiera-Large モデル パフォーマンス監視ガイド

## リアルタイム監視システム

### GPU メモリ監視
```bash
# リアルタイムGPU監視
watch -n 2 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F, "{printf \"GPU: %s\\nMemory: %d/%d MB (%.1f%%)\\nUtil: %d%%\\n\\n\", \$1, \$2, \$3, \$2/\$3*100, \$4}"'

# メモリ使用率アラート
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{if($1/$2*100 > 90) print "WARNING: GPU Memory > 90%"}'
```

### システムメモリ監視
```bash
# メモリ使用状況詳細
free -h && echo "" && cat /proc/meminfo | grep -E "MemAvailable|SwapTotal|SwapFree"

# プロセス別メモリ使用量
ps aux --sort=-%mem | head -10
```

### 学習進捗監視
```bash
# 学習ログをリアルタイム監視
tail -f external/sam2/sam2_logs/foodmix_l_*/logs/train.log | grep -E "(loss|epoch|step|memory)"
```

## パフォーマンス最適化チューニング

### 段階的メモリ削減プロセス

#### レベル1: 基本最適化
```yaml
train_batch_size: 1
max_num_objects: 25
num_train_workers: 2
pin_memory: false
amp_dtype: bfloat16
```

#### レベル2: 中程度最適化
```yaml
max_num_objects: 15
gradient_accumulation_steps: 2
resolution: 832  # 1024から削減
num_correction_pt_per_frame: 3  # デフォルト5から削減
```

#### レベル3: 積極的最適化
```yaml
max_num_objects: 10
resolution: 640
gradient_accumulation_steps: 4
num_train_workers: 1
save_freq: 500  # より頻繁な保存
```

#### レベル4: 極限最適化
```yaml
max_num_objects: 5
resolution: 512
multiplier: 1  # データ増強削減
phases_per_epoch: 1
log_freq: 200  # ログ頻度削減
```

### 速度最適化設定

#### 計算効率向上
```yaml
# 勾配クリッピング最適化
gradient_clip:
  max_norm: 0.1
  norm_type: 2

# レイヤー重み減衰最適化
layer_decay_value: 0.9

# 学習率スケジューラ軽量化
lr_scheduler: CosineParamScheduler  # より効率的
```

#### I/O最適化
```yaml
num_train_workers: 2  # 適切なワーカー数
pin_memory: false     # WSL環境では無効化
drop_last: true       # 不完全バッチスキップ
persistent_workers: true  # ワーカー再利用
```

## 異常検知とアラート

### OOM (Out of Memory) 早期警告
```bash
#!/bin/bash
# gpu_memory_alert.sh
while true; do
    USAGE=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk -F, '{print $1/$2*100}')
    if (( $(echo "$USAGE > 85" | bc -l) )); then
        echo "$(date): GPU Memory Alert: ${USAGE}%" | tee -a gpu_alerts.log
        if (( $(echo "$USAGE > 95" | bc -l) )); then
            echo "CRITICAL: GPU Memory > 95%. OOM imminent!" | tee -a gpu_alerts.log
            # Optional: 自動的に学習を一時停止
            # pkill -STOP -f train.py
        fi
    fi
    sleep 10
done
```

### 学習停滞検知
```bash
#!/bin/bash
# training_progress_monitor.sh
LOG_FILE="external/sam2/sam2_logs/foodmix_l_*/logs/train.log"
LAST_LOSS=""
STAGNANT_COUNT=0

while true; do
    CURRENT_LOSS=$(tail -20 "$LOG_FILE" | grep "loss" | tail -1 | grep -o "loss: [0-9.]*" | cut -d: -f2 | tr -d ' ')
    
    if [ "$CURRENT_LOSS" = "$LAST_LOSS" ]; then
        ((STAGNANT_COUNT++))
        if [ $STAGNANT_COUNT -gt 10 ]; then
            echo "$(date): Training may be stagnated. Same loss for $STAGNANT_COUNT checks" | tee -a training_alerts.log
        fi
    else
        STAGNANT_COUNT=0
    fi
    
    LAST_LOSS="$CURRENT_LOSS"
    sleep 60
done
```

## ベンチマーク・プロファイリング

### メモリ使用量ベンチマーク
```bash
#!/bin/bash
# memory_benchmark.sh
echo "=== SAM2.1 Large Model Memory Benchmark ==="
echo "Date: $(date)"
echo ""

# 学習前のベースライン
echo "Baseline GPU Memory:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

# 1エポックテストでメモリプロファイル
echo "Starting 1-epoch memory profiling..."
python external/sam2/training/train.py \
  -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_test.yaml \
  --use-cluster 0 --num-gpus 1 &

TRAIN_PID=$!
sleep 30  # 学習開始まで待機

echo "Peak GPU Memory during training:"
MAX_MEM=0
for i in {1..60}; do
    CURRENT_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ $CURRENT_MEM -gt $MAX_MEM ]; then
        MAX_MEM=$CURRENT_MEM
    fi
    sleep 5
done

echo "Maximum GPU Memory Used: ${MAX_MEM}MB"
wait $TRAIN_PID
```

### 速度ベンチマーク
```bash
#!/bin/bash
# speed_benchmark.sh
echo "=== SAM2.1 Large vs Base+ Speed Comparison ==="

# Base+ 1エポック時間測定
echo "Measuring Base+ training time..."
BASE_START=$(date +%s)
timeout 3600 python external/sam2/training/train.py \
  -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_test.yaml \
  --use-cluster 0 --num-gpus 1 > /dev/null 2>&1
BASE_END=$(date +%s)
BASE_TIME=$((BASE_END - BASE_START))

# Large 1エポック時間測定
echo "Measuring Large training time..."
LARGE_START=$(date +%s)
timeout 3600 python external/sam2/training/train.py \
  -c external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_test.yaml \
  --use-cluster 0 --num-gpus 1 > /dev/null 2>&1
LARGE_END=$(date +%s)
LARGE_TIME=$((LARGE_END - LARGE_START))

echo ""
echo "Results:"
echo "Base+ Time: ${BASE_TIME}s"
echo "Large Time: ${LARGE_TIME}s"
echo "Speed Ratio: $(echo "scale=2; $LARGE_TIME / $BASE_TIME" | bc)x slower"
```

## トラブルシューティング自動化

### 自動復旧スクリプト
```bash
#!/bin/bash
# auto_recovery.sh
LOG_FILE="external/sam2/sam2_logs/foodmix_l_*/logs/train.log"

while true; do
    # OOM検出
    if tail -50 "$LOG_FILE" | grep -q "out of memory\|CUDA out of memory"; then
        echo "$(date): OOM detected. Switching to memory-optimized config..."
        pkill -f train.py
        sleep 10
        bash scripts/train_with_monitoring.sh --large-model --memory-optimized --auto-resume &
    fi
    
    # 学習プロセス死活監視
    if ! pgrep -f train.py > /dev/null; then
        echo "$(date): Training process died. Attempting restart..."
        bash scripts/train_with_monitoring.sh --large-model --auto-resume &
    fi
    
    sleep 120
done
```

## パフォーマンス最適化結果の記録

### 設定・結果ログ
```bash
# performance_log.txt の自動生成
cat > performance_log.txt << EOF
SAM2.1 Large Model Performance Log
Date: $(date)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)

Configuration Used:
- batch_size: 1
- max_num_objects: $(grep max_num_objects config.yaml | awk '{print $2}')
- resolution: $(grep resolution config.yaml | awk '{print $2}')
- amp_dtype: bfloat16

Performance Metrics:
- Peak GPU Memory: TBD
- Training Speed: TBD steps/sec
- Est. Time for 40 epochs: TBD hours
EOF
```

これらの監視・最適化システムにより、Largeモデルの安定した学習実行が可能になります。