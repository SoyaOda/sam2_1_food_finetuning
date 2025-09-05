"""
メモリ最適化されたSAM2 Trainer
長時間の学習でもメモリ使用量を安定させる機能を提供
"""

import gc
import logging
import time
import psutil
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from training.trainer import Trainer
from training.utils.train_utils import AverageMeter, MemMeter


class MemoryOptimizedTrainer(Trainer):
    """メモリ監視と最適化機能付きTrainer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # メモリ監視用メータ
        self.cpu_mem_meter = AverageMeter('CPU_Mem', ':.1f')
        self.gpu_mem_meter = AverageMeter('GPU_Mem', ':.1f')
        
        # メモリクリーンアップの設定
        self.memory_cleanup_freq = getattr(self.trainer_conf, 'memory_cleanup_freq', 100)
        self.force_gc_freq = getattr(self.trainer_conf, 'force_gc_freq', 50)
        
        # メモリ使用量の閾値
        self.cpu_memory_threshold = getattr(self.trainer_conf, 'cpu_memory_threshold', 80.0)  # %
        self.gpu_memory_threshold = getattr(self.trainer_conf, 'gpu_memory_threshold', 90.0)  # %
        
        # 統計情報
        self.memory_cleanup_count = 0
        self.gc_collection_count = 0
        
        logging.info(f"Memory optimization enabled:")
        logging.info(f"  - Cleanup frequency: {self.memory_cleanup_freq} steps")
        logging.info(f"  - GC frequency: {self.force_gc_freq} steps")
        logging.info(f"  - CPU threshold: {self.cpu_memory_threshold}%")
        logging.info(f"  - GPU threshold: {self.gpu_memory_threshold}%")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """現在のメモリ使用量を取得"""
        memory_info = {}
        
        # CPUメモリ
        cpu_memory = psutil.virtual_memory()
        memory_info['cpu_used_gb'] = cpu_memory.used / 1024**3
        memory_info['cpu_total_gb'] = cpu_memory.total / 1024**3
        memory_info['cpu_percent'] = cpu_memory.percent
        
        # GPUメモリ
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory = torch.cuda.memory_stats()
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            memory_info['gpu_max_reserved_gb'] = torch.cuda.max_memory_reserved() / 1024**3
        else:
            memory_info.update({
                'gpu_allocated_gb': 0.0,
                'gpu_reserved_gb': 0.0,
                'gpu_max_reserved_gb': 0.0
            })
        
        return memory_info
    
    def _cleanup_memory(self, force: bool = False) -> bool:
        """メモリクリーンアップを実行"""
        memory_before = self._get_memory_usage()
        
        # 条件チェック
        should_cleanup = (
            force or
            memory_before['cpu_percent'] > self.cpu_memory_threshold or
            (torch.cuda.is_available() and 
             memory_before['gpu_reserved_gb'] > 0 and
             (memory_before['gpu_allocated_gb'] / memory_before['gpu_reserved_gb']) * 100 > self.gpu_memory_threshold)
        )
        
        if not should_cleanup:
            return False
        
        # クリーンアップ実行
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Pythonガベージコレクション
        collected = gc.collect()
        self.gc_collection_count += collected
        
        memory_after = self._get_memory_usage()
        self.memory_cleanup_count += 1
        
        # ログ出力
        cpu_saved = memory_before['cpu_used_gb'] - memory_after['cpu_used_gb']
        gpu_saved = memory_before['gpu_reserved_gb'] - memory_after['gpu_reserved_gb']
        
        logging.info(f"Memory cleanup #{self.memory_cleanup_count}:")
        logging.info(f"  CPU: {memory_before['cpu_used_gb']:.1f}GB → {memory_after['cpu_used_gb']:.1f}GB (saved: {cpu_saved:.1f}GB)")
        logging.info(f"  GPU: {memory_before['gpu_reserved_gb']:.1f}GB → {memory_after['gpu_reserved_gb']:.1f}GB (saved: {gpu_saved:.1f}GB)")
        logging.info(f"  GC objects collected: {collected}")
        
        return True
    
    def _log_memory_usage(self, prefix: str = ""):
        """メモリ使用量をログに出力"""
        memory_info = self._get_memory_usage()
        
        self.cpu_mem_meter.update(memory_info['cpu_used_gb'])
        self.gpu_mem_meter.update(memory_info['gpu_allocated_gb'])
        
        log_msg = f"{prefix}Memory Usage: "
        log_msg += f"CPU: {memory_info['cpu_used_gb']:.1f}/{memory_info['cpu_total_gb']:.1f}GB ({memory_info['cpu_percent']:.1f}%), "
        log_msg += f"GPU: {memory_info['gpu_allocated_gb']:.1f}/{memory_info['gpu_reserved_gb']:.1f}GB"
        
        if memory_info['gpu_reserved_gb'] > 0:
            gpu_utilization = (memory_info['gpu_allocated_gb'] / memory_info['gpu_reserved_gb']) * 100
            log_msg += f" ({gpu_utilization:.1f}%)"
        
        logging.info(log_msg)
    
    def _train_one_epoch(self, epoch: int):
        """メモリ最適化版のエポック学習"""
        
        # 開始時のメモリ状況をログ
        self._log_memory_usage(f"Epoch {epoch} start - ")
        
        # 元の学習ループをオーバーライド
        self.model.train()
        
        for i, batch in enumerate(self.train_loader):
            # 定期的なメモリクリーンアップ
            if i % self.memory_cleanup_freq == 0 and i > 0:
                self._cleanup_memory(force=False)
            
            # 強制ガベージコレクション
            if i % self.force_gc_freq == 0 and i > 0:
                gc.collect()
            
            # バッチ処理
            self._train_step(batch, i, epoch)
            
            # メモリ使用量のモニタリング
            if i % (self.logging_conf.log_freq * 5) == 0:
                self._log_memory_usage(f"Step {self.steps} - ")
        
        # エポック終了時のクリーンアップ
        self._cleanup_memory(force=True)
        self._log_memory_usage(f"Epoch {epoch} end - ")
        
        logging.info(f"Epoch {epoch} memory statistics:")
        logging.info(f"  Total cleanups: {self.memory_cleanup_count}")
        logging.info(f"  GC collections: {self.gc_collection_count}")
        logging.info(f"  Avg CPU usage: {self.cpu_mem_meter.avg:.1f}GB")
        logging.info(f"  Avg GPU usage: {self.gpu_mem_meter.avg:.1f}GB")
    
    def _train_step(self, batch: Any, step_idx: int, epoch: int):
        """メモリ効率的な学習ステップ"""
        # 既存のtrain_stepロジックを呼び出し
        # 実装は親クラスのメソッドに依存
        
        batch_start_time = time.time()
        
        # バッチをデバイスに移動
        if hasattr(batch, 'to'):
            batch = batch.to(self.device, non_blocking=True)
        
        # 勾配リセット
        self.optim.optimizer.zero_grad(set_to_none=True)
        
        # フォワードパス
        with torch.cuda.amp.autocast(enabled=self.optim_conf.amp.enabled):
            outputs = self.model(batch)
            loss_dict = self.loss(outputs)
            loss = loss_dict['loss']
        
        # バックワードパス
        if self.optim_conf.amp.enabled:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optim.optimizer.step()
        
        # 学習率スケジューリング
        self.optim.scheduler_step()
        
        # メトリクス更新
        batch_time = time.time() - batch_start_time
        
        # ステップカウンタ更新
        self.steps += 1
        
        # ログ出力
        if step_idx % self.logging_conf.log_freq == 0:
            self._log_training_metrics(loss_dict, batch_time, step_idx, epoch)
    
    def _log_training_metrics(self, loss_dict: Dict[str, torch.Tensor], batch_time: float, step_idx: int, epoch: int):
        """学習メトリクスのログ出力（メモリ情報を含む）"""
        memory_info = self._get_memory_usage()
        
        log_msg = f"Train Epoch: [{epoch}][{step_idx}/{len(self.train_loader)}] | "
        log_msg += f"Batch Time: {batch_time:.2f}s | "
        log_msg += f"GPU Mem: {memory_info['gpu_allocated_gb']:.1f}GB | "
        log_msg += f"CPU Mem: {memory_info['cpu_percent']:.1f}% | "
        
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                log_msg += f"{key}: {value.item():.2e} | "
        
        logging.info(log_msg.rstrip(" | "))