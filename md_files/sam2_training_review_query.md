# SAM2.1 食品セグメンテーション学習システム レビュークエリ

## 概要
SAM2.1を使用した食品画像セグメンテーションのファインチューニングシステムを実装し、15エポックの学習を完了しました。実装コード、設定、学習経過について専門的なレビューとさらなる改善提案を求めます。

## 1. メイン学習スクリプト

```python
#!/usr/bin/env python3
"""
Memory-optimized SAM2.1 Training Script
メモリ効率を重視したSAM2.1学習スクリプト
"""

import os
import sys
import gc
import logging
import argparse
import psutil
import time
from contextlib import contextmanager

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import register_new_resolver

# WandBサポート（オプション）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("WandB not available. Install with: pip install wandb")


@contextmanager
def MemoryMonitoringContext(operation_name: str):
    """メモリ監視付きコンテキストマネージャ"""
    
    def get_memory_info():
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            gpu_memory = 0.0
        
        cpu_memory = psutil.virtual_memory().used / (1024**3)  # GB
        cpu_percent = psutil.virtual_memory().percent
        
        return cpu_memory, cpu_percent, gpu_memory
    
    cpu_start, cpu_percent_start, gpu_start = get_memory_info()
    start_time = time.time()
    
    logging.info(f"{operation_name} - Start: CPU {cpu_start:.1f}GB ({cpu_percent_start:.1f}%), GPU {gpu_start:.1f}GB")
    
    try:
        yield
    finally:
        end_time = time.time()
        cpu_end, cpu_percent_end, gpu_end = get_memory_info()
        duration = end_time - start_time
        
        logging.info(f"{operation_name} - End: CPU {cpu_end:.1f}GB ({cpu_percent_end:.1f}%), GPU {gpu_end:.1f}GB")
        logging.info(f"{operation_name} - Duration: {duration:.1f}s")
        logging.info(f"{operation_name} - Change: CPU {cpu_end-cpu_start:+.1f}GB, GPU {gpu_end-gpu_start:+.1f}GB")


def setup_memory_monitoring():
    """メモリ監視の初期設定"""
    
    if torch.cuda.is_available():
        # CUDAメモリの初期化とクリア
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # CUDAキャッシュ最適化
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        logging.info("CUDA optimization settings applied")
    
    # CPUスレッド数の最適化
    cpu_count = psutil.cpu_count(logical=False)  # 物理コア数
    torch.set_num_threads(min(cpu_count, 4))  # 過度な並列化を避ける
    
    logging.info(f"PyTorch threads set to: {torch.get_num_threads()}")


def optimize_pytorch_settings():
    """PyTorchの最適化設定"""
    
    # メモリ効率化設定
    torch.backends.cudnn.deterministic = False  # 速度優先
    torch.backends.cudnn.benchmark = True      # 最適化されたアルゴリズム選択
    
    # 自動混合精度の設定確認
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] >= 7:  # Tensor Cores利用可能
            logging.info(f"Tensor Cores available on device (compute capability: {device_cap})")
        else:
            logging.info(f"Tensor Cores not available (compute capability: {device_cap})")


def periodic_memory_cleanup():
    """定期的なメモリクリーンアップ"""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    collected = gc.collect()
    
    if collected > 0:
        logging.info(f"Garbage collection: {collected} objects collected")


def check_system_stability() -> bool:
    """システムの安定性をチェック"""
    
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # メモリ使用率が95%を超えた場合
    if memory.percent > 95.0:
        logging.error(f"Critical CPU memory usage: {memory.percent:.1f}%")
        return False
    
    # CPU使用率が持続的に高い場合
    if cpu_percent > 90.0:
        logging.warning(f"High CPU usage: {cpu_percent:.1f}%")
    
    return True


def register_omegaconf_resolvers():
    """OmegaConfのカスタムリゾルバを登録"""
    
    try:
        register_new_resolver("times", lambda x, y: int(x) * int(y))
        register_new_resolver("divide", lambda x, y: float(x) / float(y))
        logging.info("OmegaConf resolvers registered successfully")
    except Exception as e:
        logging.warning(f"Failed to register OmegaConf resolvers: {e}")


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized SAM2.1 Training")
    parser.add_argument(
        "-c", "--config", 
        required=True,
        help="Config name (e.g., sam2.1_training/sam2.1_hiera_b+_foodmix_optimized)"
    )
    parser.add_argument("--use-cluster", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--memory-check-interval", type=int, default=50, 
                       help="Memory check interval in steps")
    parser.add_argument("--auto-restart-on-oom", action='store_true',
                       help="Automatically restart training on OOM")
    # WandB引数
    parser.add_argument("--use-wandb", action='store_true',
                       help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="sam2_food_segmentation",
                       help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="WandB entity (username/team)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs to train (overrides config)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--max-epochs", type=int, default=None,
                       help="Maximum epochs (for continuing training)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--save-freq", type=int, default=None,
                       help="Checkpoint save frequency (overrides config)")
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{args.config.replace("/", "_")}.log')
        ]
    )
    
    logging.info("="*80)
    logging.info("MEMORY-OPTIMIZED SAM2.1 TRAINING")
    logging.info("="*80)
    
    # WandB初期化
    use_wandb = False
    if args.use_wandb and WANDB_AVAILABLE:
        try:
            # WandBのAPIキーが設定されていない場合は警告してWandBを無効化
            if 'WANDB_API_KEY' not in os.environ:
                logging.warning("WANDB_API_KEY not set in environment variables")
                logging.warning("WandB tracking disabled, continuing with local logging only")
                use_wandb = False
            else:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config=vars(args),
                    name=f"sam2_food_{args.config.replace('/', '_')}"
                )
            use_wandb = True
            logging.info(f"WandB initialized: {args.wandb_project}")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB: {e}")
            use_wandb = False
    elif args.use_wandb and not WANDB_AVAILABLE:
        logging.warning("WandB requested but not available. Install with: pip install wandb")
    
    # システム状態の確認
    with MemoryMonitoringContext("System Setup"):
        setup_memory_monitoring()
        optimize_pytorch_settings()
    
    # Hydraをリセット
    GlobalHydra.instance().clear()
    
    # カスタム初期化
    config_dir = os.path.abspath("sam2/configs")
    
    with MemoryMonitoringContext("Hydra Initialization"):
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        
        # OmegaConfリゾルバの登録
        register_omegaconf_resolvers()
        
        # 設定を読み込み
        cfg = compose(config_name=args.config)
    
    # メモリ監視設定をconfigに追加（OmegaConfのstruct制約を回避）
    from omegaconf import OmegaConf
    
    # structモードを一時的に無効化
    OmegaConf.set_struct(cfg, False)
    
    # メモリ監視設定を追加
    cfg.memory_optimization = OmegaConf.create({
        'cleanup_freq': 25,  # 25ステップごとにクリーンアップ
        'gc_freq': 10,       # 10ステップごとにGC
        'cpu_threshold': 80.0,
        'gpu_threshold': 85.0  # より厳格な閾値
    })
    
    # コマンドライン引数で設定をオーバーライド
    if args.epochs is not None:
        cfg.scratch.num_epochs = args.epochs
        logging.info(f"Epochs overridden to: {args.epochs}")
    
    if args.lr is not None:
        cfg.scratch.base_lr = args.lr
        cfg.scratch.vision_lr = args.lr * 0.6  # vision encoderはより低い学習率
        logging.info(f"Learning rate overridden to: {args.lr}")
    
    if args.save_freq is not None:
        cfg.trainer.checkpoint.save_freq = args.save_freq
        logging.info(f"Save frequency overridden to: {args.save_freq}")
    
    # Resume from checkpoint handling
    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            cfg.trainer.checkpoint.resume_from = args.resume_from
            logging.info(f"Will resume from checkpoint: {args.resume_from}")
        else:
            logging.error(f"Resume checkpoint not found: {args.resume_from}")
            return
    
    # structモードを再有効化
    OmegaConf.set_struct(cfg, True)
    
    # argsを設定に追加
    args.use_cluster = bool(args.use_cluster)
    
    # SAM2のTrainerクラスにカスタムロガーを適用
    def patch_trainer_logger():
        """TrainerクラスのLogger初期化をカスタムロガーでパッチ"""
        from training.trainer import Trainer
        
        # 元のTrainerの__init__メソッドを保存
        original_init = Trainer.__init__
        
        def patched_init(self, *args, **kwargs):
            # 元の初期化を実行
            original_init(self, *args, **kwargs)
            
            # カスタムメモリ監視を追加
            original_train_step = self.run_step
            
            def memory_monitored_step():
                if hasattr(self, '_step_count'):
                    self._step_count += 1
                else:
                    self._step_count = 0
                
                # 定期的なメモリクリーンアップ
                if self._step_count % 25 == 0:
                    periodic_memory_cleanup()
                
                return original_train_step()
            
            self.run_step = memory_monitored_step
        
        # Trainerクラスにパッチをあてる
        Trainer.__init__ = patched_init
    
    # パッチを適用
    patch_trainer_logger()
    
    # train.pyのmain関数を呼び出し
    from training.train import single_node_runner
    import random
    
    # ログディレクトリの設定
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config.replace("/", "_") + "_memory_optimized"
        )
    
    # ディレクトリ作成
    os.makedirs(cfg.launcher.experiment_log_dir, exist_ok=True)
    
    # 設定を保存
    from omegaconf import OmegaConf
    config_path = os.path.join(cfg.launcher.experiment_log_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # submitit設定
    if "submitit" not in cfg:
        cfg.submitit = OmegaConf.create({
            "port_range": [10000, 65000]
        })
    
    # ランダムポート選択
    main_port = random.randint(
        cfg.submitit.port_range[0], cfg.submitit.port_range[1]
    )
    
    # 実行パラメータの設定
    cfg.launcher.num_nodes = 1
    cfg.launcher.gpus_per_node = args.num_gpus
    
    # 学習情報の表示
    logging.info("TRAINING CONFIGURATION:")
    logging.info(f"  Config: {args.config}")
    logging.info(f"  Experiment dir: {cfg.launcher.experiment_log_dir}")
    logging.info(f"  Dataset images: {cfg.dataset.img_folder}")
    logging.info(f"  Dataset annotations: {cfg.dataset.gt_folder}")
    logging.info(f"  Training list: {cfg.dataset.file_list_txt}")
    logging.info(f"  Batch size: {cfg.scratch.train_batch_size}")
    logging.info(f"  Epochs: {cfg.scratch.num_epochs}")
    logging.info(f"  Learning rate: {cfg.scratch.base_lr}")
    logging.info(f"  Memory check interval: {args.memory_check_interval} steps")
    logging.info("="*80)
    
    # 最終メモリ状況確認
    if not check_system_stability():
        logging.error("System instability detected. Aborting training.")
        sys.exit(1)
    
    # 学習実行
    try:
        with MemoryMonitoringContext("Training Execution"):
            single_node_runner(cfg, main_port)
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error(f"GPU Out of Memory Error: {e}")
            logging.error("Try reducing batch_size, resolution, or num_workers")
            
            if args.auto_restart_on_oom:
                logging.info("Attempting to restart with reduced settings...")
                # 設定を調整して再試行
                cfg.scratch.train_batch_size = max(1, cfg.scratch.train_batch_size // 2)
                cfg.scratch.num_train_workers = max(1, cfg.scratch.num_train_workers // 2)
                logging.info(f"Reduced batch_size to {cfg.scratch.train_batch_size}")
                logging.info(f"Reduced num_workers to {cfg.scratch.num_train_workers}")
                
                # 再実行
                single_node_runner(cfg, main_port)
            else:
                sys.exit(1)
        else:
            logging.error(f"Training error: {e}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during training: {e}")
        sys.exit(1)
    finally:
        # WandBクリーンアップ
        if use_wandb:
            try:
                wandb.finish()
            except:
                pass
        
        # 最終メモリクリーンアップ
        periodic_memory_cleanup()
        
        final_memory = psutil.virtual_memory().percent
        logging.info(f"Final system memory usage: {final_memory:.1f}%")


if __name__ == "__main__":
    main()
```

## 2. 学習設定ファイル（sam2.1_hiera_b+_foodmix_optimized.yaml）

```yaml
# @package _global_
# メモリ最適化版SAM2.1設定

scratch:
  resolution: 1024  # spec1.md準拠: 1024×1024固定
  train_batch_size: 1  # spec1.md準拠: 単GPU/T4用
  num_train_workers: 2  # WSL環境に適したワーカー数
  num_frames: 1
  max_num_objects: 3  # 公式推奨値: メモリ効率と安定性重視
  base_lr: 5.0e-6  # 公式推奨値: 低めの学習率で安定化
  vision_lr: 3.0e-6  # 公式推奨値: さらに低めで視覚エンコーダ
  phases_per_epoch: 1
  num_epochs: 40  # spec1.md準拠: 40エポック

dataset:
  img_folder: /home/soya/sam2_1_food_finetuning/data/foodmix_sa1b/images
  gt_folder: /home/soya/sam2_1_food_finetuning/data/foodmix_sa1b/annotations
  file_list_txt: /home/soya/sam2_1_food_finetuning/data/foodmix_sa1b_splits/train.txt
  multiplier: 1  # メモリ使用量削減

# ... [trainer, model, optimizer etc. の詳細設定]
trainer:
  max_epochs: ${times:${scratch.num_epochs},${scratch.phases_per_epoch}}
  accelerator: cuda
  precision: "bf16"       # A100等。T4なら "fp32" に変更
  
  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 200   # 頻繁なチェックポイント保存
    
  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16  # 公式推奨: より安定した混合精度
      
    gradient_clip:
      max_norm: 0.1
      norm_type: 2
```

## 3. 学習経過データ

### 学習統計（15エポック完了）
```json
{
  "学習時間": "11時間53分",
  "総ステップ数": 168555,
  "エポックあたりステップ": 11237,
  
  "損失推移": {
    "epoch_0": {"総合Loss": 1.374, "マスクLoss": 0.034, "DiceLoss": 0.467, "IoULoss": 0.225},
    "epoch_5": {"総合Loss": 1.121, "マスクLoss": 0.027, "DiceLoss": 0.402, "IoULoss": 0.177},
    "epoch_10": {"総合Loss": 0.876, "マスクLoss": 0.020, "DiceLoss": 0.334, "IoULoss": 0.139},
    "epoch_14": {"総合Loss": 0.797, "マスクLoss": 0.019, "DiceLoss": 0.303, "IoULoss": 0.123}
  },
  
  "改善率": {
    "総合Loss": "42%改善 (1.374→0.797)",
    "マスクLoss": "46%改善 (0.034→0.019)", 
    "DiceLoss": "35%改善 (0.467→0.303)",
    "IoULoss": "45%改善 (0.225→0.123)"
  },
  
  "システム効率": {
    "GPU使用率": "安定29.2%",
    "CPUメモリ": "平均22%", 
    "バッチ処理時間": "0.20-0.22秒/バッチ",
    "メモリリーク": "なし"
  }
}
```

### システム監視ログサンプル
```
INFO 2025-09-07 23:05:04 Train Epoch: [14][9100/11237] | Batch Time: 0.21 (0.22) | Mem (GB): 4.00 (4.65/5.00) | Losses/train_all_loss: 1.77e-01 (8.01e-01)
INFO 2025-09-07 23:12:35 Losses and meters: {'Losses/train_all_loss': 0.7968479934962124, 'Trainer/epoch': 14, 'Trainer/steps_train': 168555}
INFO 2025-09-07 23:12:37 Training Execution - End: CPU 5.0GB (17.3%), GPU 0.3GB
```

## 4. 実装特徴

### メモリ最適化
- **MemoryMonitoringContext**: 各処理のメモリ使用量を詳細追跡
- **定期的ガベージコレクション**: 25ステップごとのメモリクリーンアップ
- **CUDA最適化**: TensorCore活用、キャッシュ最適化
- **OOM自動復旧**: メモリ不足時の自動設定調整・再開

### 学習継続システム
- **チェックポイント自動検出**: 最新実験の自動継続
- **パラメータオーバーライド**: コマンドライン引数で柔軟な設定変更
- **WandB統合**: API未設定でも継続動作
- **システム監視**: リアルタイムリソース監視

### データセット構成
- **FoodSeg103**: 材料レベルのセグメンテーション
- **UEC-FoodPix**: 料理レベルのセグメンテーション
- **SA-1B形式**: SAM2.1ネイティブ形式に正規化済み
- **1024×1024解像度**: spec1.md準拠の固定解像度

## 5. 質問・レビュー観点

### コード品質
1. **メモリ管理**: 現在の実装で見落としている最適化ポイントはありますか？
2. **エラーハンドリング**: より堅牢なエラー処理の改善点は？
3. **設計パターン**: コードアーキテクチャの改善提案は？

### 学習効率
1. **ハイパーパラメータ**: 現在の設定値（学習率5e-6、バッチサイズ1等）の妥当性は？
2. **学習曲線**: Loss減少パターンから見た最適化余地は？
3. **収束性**: さらなる学習継続の効果予測は？

### システム設計
1. **スケーラビリティ**: マルチGPU対応時の課題は？
2. **再現性**: 実験再現性の担保方法は適切ですか？
3. **監視体制**: 追加すべきメトリクスは？

### 次ステップ提案
1. **性能評価**: 定量的評価手法の推奨は？
2. **データ拡張**: さらなる精度向上のためのデータ戦略は？
3. **実用化**: プロダクション環境への展開時の注意点は？

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。