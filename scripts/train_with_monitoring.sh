#!/bin/bash
#
# メモリ監視付き学習実行スクリプト
# システムリソースを監視し、異常時にアラートを出す
#

# 設定
CONFIG_NAME="sam2.1_training/sam2.1_hiera_b+_foodmix_optimized"
MEMORY_THRESHOLD=90  # CPUメモリ使用率の閾値（％）
GPU_MEMORY_THRESHOLD=80  # より厳しく設定  # GPUメモリ使用率の閾値（％）
CHECK_INTERVAL=30  # 監視間隔（秒）
LOG_FILE="training_monitor.log"

# 色付きログ関数
log_info() {
    echo -e "\033[32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

# システム情報を取得
get_system_info() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    local memory_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    local gpu_memory_usage=""
    
    if command -v nvidia-smi &> /dev/null; then
        gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk -F', ' '{printf("%.1f", $1/$2*100)}')
    fi
    
    echo "$cpu_usage,$memory_usage,$gpu_memory_usage"
}

# システム監視
monitor_system() {
    local training_pid=$1
    
    log_info "Starting system monitoring (PID: $training_pid)"
    
    while kill -0 "$training_pid" 2>/dev/null; do
        local info=$(get_system_info)
        local cpu_usage=$(echo "$info" | cut -d',' -f1)
        local memory_usage=$(echo "$info" | cut -d',' -f2)
        local gpu_memory_usage=$(echo "$info" | cut -d',' -f3)
        
        # CPU使用率チェック
        if (( $(echo "$cpu_usage > 85" | bc -l) )); then
            log_warn "High CPU usage: ${cpu_usage}%"
        fi
        
        # メモリ使用率チェック
        if (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
            log_warn "High memory usage: ${memory_usage}%"
            
            if (( $(echo "$memory_usage > 98" | bc -l) )); then
                log_error "Critical memory usage: ${memory_usage}%. Consider stopping training."
            fi
        fi
        
        # GPUメモリ使用率チェック
        if [[ -n "$gpu_memory_usage" ]] && (( $(echo "$gpu_memory_usage > $GPU_MEMORY_THRESHOLD" | bc -l) )); then
            log_warn "High GPU memory usage: ${gpu_memory_usage}%"
        fi
        
        # 定期的な状況報告
        log_info "System status: CPU ${cpu_usage}%, Memory ${memory_usage}%, GPU Memory ${gpu_memory_usage}%"
        
        sleep "$CHECK_INTERVAL"
    done
    
    log_info "Training process ended. Monitoring stopped."
}

# クリーンアップ関数
cleanup() {
    log_info "Cleaning up..."
    
    # Pythonプロセスの終了
    if [[ -n "$TRAINING_PID" ]]; then
        log_info "Terminating training process (PID: $TRAINING_PID)"
        kill -TERM "$TRAINING_PID" 2>/dev/null
        sleep 5
        kill -KILL "$TRAINING_PID" 2>/dev/null
    fi
    
    # 監視プロセスの終了
    if [[ -n "$MONITOR_PID" ]]; then
        kill -TERM "$MONITOR_PID" 2>/dev/null
    fi
    
    log_info "Cleanup completed"
    exit 0
}

# シグナルハンドラ
trap cleanup SIGINT SIGTERM

# メイン処理
main() {
    log_info "=================================="
    log_info "SAM2.1 Training with Monitoring"
    log_info "=================================="
    
    # 引数の解析
    local use_memory_optimized=false
    local config_name="$CONFIG_NAME"
    local resume_dir=""
    local auto_resume=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --memory-optimized)
                use_memory_optimized=true
                shift
                ;;
            --config|-c)
                config_name="$2"
                shift 2
                ;;
            --resume)
                resume_dir="$2"
                shift 2
                ;;
            --auto-resume)
                auto_resume=true
                shift
                ;;
            --memory-threshold)
                MEMORY_THRESHOLD="$2"
                shift 2
                ;;
            --gpu-memory-threshold)
                GPU_MEMORY_THRESHOLD="$2"
                shift 2
                ;;
            --check-interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --memory-optimized     Use memory-optimized training script"
                echo "  --config,-c CONFIG     Training configuration name"
                echo "  --resume DIR           Resume from experiment directory"
                echo "  --auto-resume          Automatically resume from latest experiment"
                echo "  --memory-threshold N   CPU memory threshold % (default: 90)"
                echo "  --gpu-memory-threshold N GPU memory threshold % (default: 80)"
                echo "  --check-interval N     Monitoring interval in seconds (default: 30)"
                echo "  --help,-h             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # システム情報を表示
    log_info "System Information:"
    log_info "  OS: $(uname -s) $(uname -r)"
    log_info "  CPU cores: $(nproc)"
    log_info "  Memory: $(free -h | grep Mem | awk '{print $2}') total"
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
        log_info "  GPU: $gpu_info"
    fi
    
    log_info "Monitoring Configuration:"
    log_info "  Memory threshold: ${MEMORY_THRESHOLD}%"
    log_info "  GPU memory threshold: ${GPU_MEMORY_THRESHOLD}%"
    log_info "  Check interval: ${CHECK_INTERVAL}s"
    
    # 自動再開機能のチェック
    if [[ "$auto_resume" == true ]] && [[ -z "$resume_dir" ]]; then
        # 最新の実験ディレクトリを自動検出
        if [[ -d "./sam2_logs" ]]; then
            log_info "Searching for resumable experiments..."
            local latest_experiment=""
            local latest_time=0
            
            for exp_dir in ./sam2_logs/*/; do
                if [[ -d "$exp_dir/checkpoints" ]] && ls "$exp_dir/checkpoints"/*.pt >/dev/null 2>&1; then
                    local exp_time=$(stat -c %Y "$exp_dir" 2>/dev/null || echo 0)
                    if [[ $exp_time -gt $latest_time ]]; then
                        latest_time=$exp_time
                        latest_experiment="$exp_dir"
                    fi
                fi
            done
            
            if [[ -n "$latest_experiment" ]]; then
                resume_dir="${latest_experiment%/}"  # Remove trailing slash
                log_info "Auto-detected latest experiment: $(basename "$resume_dir")"
            else
                log_info "No resumable experiments found. Starting new training."
            fi
        fi
    fi
    
    # レジューム処理
    if [[ -n "$resume_dir" ]]; then
        log_info "Resuming training from: $resume_dir"
        
        local script_args="--experiment-dir \"$resume_dir\""
        if [[ "$use_memory_optimized" == true ]]; then
            script_args="$script_args --memory-optimized"
        fi
        
        python scripts/resume_training.py $script_args &
        TRAINING_PID=$!
    else
        # 新規学習開始
        log_info "Starting new training with config: $config_name"
        
        local training_script="scripts/train_sam2_wrapper.py"
        if [[ "$use_memory_optimized" == true ]]; then
            training_script="scripts/train_sam2_memory_optimized.py"
            log_info "Using memory-optimized training script"
        fi
        
        python "$training_script" -c "$config_name" --use-cluster 0 --num-gpus 1 &
        TRAINING_PID=$!
    fi
    
    # 学習プロセスの確認
    if ! kill -0 "$TRAINING_PID" 2>/dev/null; then
        log_error "Failed to start training process"
        exit 1
    fi
    
    log_info "Training process started (PID: $TRAINING_PID)"
    
    # バックグラウンドで監視開始
    monitor_system "$TRAINING_PID" &
    MONITOR_PID=$!
    
    # 学習プロセスの完了を待機
    wait "$TRAINING_PID"
    local training_exit_code=$?
    
    # 監視プロセスの終了
    if [[ -n "$MONITOR_PID" ]]; then
        kill -TERM "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
    fi
    
    # 結果の報告
    if [[ $training_exit_code -eq 0 ]]; then
        log_info "Training completed successfully!"
    else
        log_error "Training failed with exit code: $training_exit_code"
    fi
    
    # 最終システム状況
    local final_info=$(get_system_info)
    local final_memory=$(echo "$final_info" | cut -d',' -f2)
    log_info "Final system memory usage: ${final_memory}%"
    
    exit $training_exit_code
}

# 実行
main "$@"