#!/usr/bin/env python3
"""
メモリ最適化版SAM2.1学習スクリプト
長時間の学習でもシステムの安定性を保つ機能を提供
"""

import os
import sys
import argparse
import logging
import psutil
import torch
import gc
from typing import Optional

# SAM2のパスを追加
sam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "sam2"))
sys.path.insert(0, sam2_path)
sys.path.insert(0, os.path.dirname(__file__) + "/..")
os.chdir(sam2_path)

# Hydraの初期化を上書き
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from training.utils.train_utils import register_omegaconf_resolvers


def setup_memory_monitoring():
    """メモリ監視の設定"""
    
    # システムメモリ情報を取得
    memory = psutil.virtual_memory()
    available_cpu_gb = memory.available / 1024**3
    total_cpu_gb = memory.total / 1024**3
    
    logging.info("="*60)
    logging.info("SYSTEM MEMORY INFO")
    logging.info("="*60)
    logging.info(f"Total CPU Memory: {total_cpu_gb:.1f} GB")
    logging.info(f"Available CPU Memory: {available_cpu_gb:.1f} GB")
    logging.info(f"CPU Memory Usage: {memory.percent:.1f}%")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Total GPU Memory: {gpu_memory:.1f} GB")
        torch.cuda.empty_cache()
        logging.info(f"GPU Memory after cache clear: {torch.cuda.memory_allocated()/1024**3:.1f} GB allocated")
    
    logging.info("="*60)
    
    # メモリが少ない場合の警告
    if available_cpu_gb < 8.0:
        logging.warning(f"⚠️  Low available CPU memory: {available_cpu_gb:.1f} GB")
        logging.warning("⚠️  Consider closing other applications")
    
    if torch.cuda.is_available() and gpu_memory < 12.0:
        logging.warning(f"⚠️  Limited GPU memory: {gpu_memory:.1f} GB")
        logging.warning("⚠️  Consider using smaller batch size or gradient checkpointing")


def optimize_pytorch_settings():
    """PyTorchのメモリ最適化設定"""
    
    # CUDAメモリ最適化
    if torch.cuda.is_available():
        # メモリフラグメンテーション削減（より厳しく設定）
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.6'
        
        # CUDAキャッシュ最適化
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        logging.info("CUDA optimization settings applied")
    
    # CPUスレッド数の最適化
    cpu_count = psutil.cpu_count(logical=False)  # 物理コア数
    torch.set_num_threads(min(cpu_count, 4))  # 過度な並列化を避ける
    
    logging.info(f"PyTorch threads set to: {torch.get_num_threads()}")


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
        logging.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
    
    return True


class MemoryMonitoringContext:
    """メモリ監視のコンテキストマネージャ"""
    
    def __init__(self, name: str = "Memory Monitor"):
        self.name = name
        self.start_memory = None
        
    def __enter__(self):
        self.start_memory = self._get_memory_info()
        logging.info(f"{self.name} - Start: {self._format_memory_info(self.start_memory)}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = self._get_memory_info()
        logging.info(f"{self.name} - End: {self._format_memory_info(end_memory)}")
        
        if self.start_memory:
            cpu_diff = end_memory['cpu_used'] - self.start_memory['cpu_used']
            gpu_diff = end_memory['gpu_allocated'] - self.start_memory['gpu_allocated']
            logging.info(f"{self.name} - Change: CPU {cpu_diff:+.1f}GB, GPU {gpu_diff:+.1f}GB")
    
    def _get_memory_info(self):
        memory = psutil.virtual_memory()
        info = {
            'cpu_used': memory.used / 1024**3,
            'cpu_percent': memory.percent,
            'gpu_allocated': 0.0,
            'gpu_reserved': 0.0
        }
        
        if torch.cuda.is_available():
            info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        
        return info
    
    def _format_memory_info(self, memory_info):
        return (f"CPU {memory_info['cpu_used']:.1f}GB ({memory_info['cpu_percent']:.1f}%), "
                f"GPU {memory_info['gpu_allocated']:.1f}GB")


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
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('memory_optimized_training.log')
        ]
    )
    
    logging.info("="*80)
    logging.info("MEMORY-OPTIMIZED SAM2.1 TRAINING")
    logging.info("="*80)
    
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
    
    # structモードを再有効化
    OmegaConf.set_struct(cfg, True)
    
    # argsを設定に追加
    args.use_cluster = bool(args.use_cluster)
    
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
            logging.error(f"Training failed with error: {e}")
            raise
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(0)
    
    finally:
        # 最終クリーンアップ
        periodic_memory_cleanup()
        final_memory = psutil.virtual_memory()
        logging.info(f"Final system memory usage: {final_memory.percent:.1f}%")


if __name__ == "__main__":
    main()