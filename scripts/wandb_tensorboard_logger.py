#!/usr/bin/env python3
"""
WandB + TensorBoard統合ロガー
SAM2.1学習でWandBとTensorBoard両方にメトリクスを記録
"""

import logging
from typing import Any, Dict, Optional, Union

from numpy import ndarray
from torch import Tensor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from training.utils.logger import Logger as OriginalLogger, TensorBoardLogger

Scalar = Union[Tensor, ndarray, int, float]


class WandBTensorBoardLogger(OriginalLogger):
    """
    TensorBoardとWandB両方をサポートする拡張Logger
    """
    
    def __init__(self, logging_conf, use_wandb: bool = False):
        # 元のTensorBoardロガーを初期化
        super().__init__(logging_conf)
        
        # WandBの使用可否を設定
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and not WANDB_AVAILABLE:
            logging.warning("WandB requested but not available. Only TensorBoard will be used.")
        
        if self.use_wandb:
            logging.info("WandB + TensorBoard logging enabled")
        else:
            logging.info("TensorBoard only logging enabled")
    
    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """メトリクスをTensorBoardとWandB両方に記録"""
        
        # TensorBoardに記録（既存機能）
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        
        # WandBに記録
        if self.use_wandb:
            try:
                # WandBではstepキーを追加
                wandb_payload = payload.copy()
                wandb_payload['step'] = step
                
                # Tensorからfloatに変換
                cleaned_payload = {}
                for k, v in wandb_payload.items():
                    if isinstance(v, Tensor):
                        cleaned_payload[k] = float(v.detach().cpu())
                    elif isinstance(v, ndarray):
                        cleaned_payload[k] = float(v)
                    else:
                        cleaned_payload[k] = float(v)
                
                wandb.log(cleaned_payload, step=step)
                
            except Exception as e:
                logging.warning(f"Failed to log to WandB: {e}")
    
    def log(self, name: str, data: Scalar, step: int) -> None:
        """単一メトリクスをTensorBoardとWandB両方に記録"""
        
        # TensorBoardに記録（既存機能）
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        
        # WandBに記録
        if self.use_wandb:
            try:
                # Tensorからfloatに変換
                if isinstance(data, Tensor):
                    value = float(data.detach().cpu())
                elif isinstance(data, ndarray):
                    value = float(data)
                else:
                    value = float(data)
                
                wandb.log({name: value, 'step': step}, step=step)
                
            except Exception as e:
                logging.warning(f"Failed to log {name} to WandB: {e}")
    
    def log_hparams(self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]) -> None:
        """ハイパーパラメータをTensorBoardとWandB両方に記録"""
        
        # TensorBoardに記録（既存機能）
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        
        # WandBに記録
        if self.use_wandb:
            try:
                # ハイパーパラメータをWandBのconfigに追加
                cleaned_hparams = {}
                for k, v in hparams.items():
                    if isinstance(v, (Tensor, ndarray)):
                        cleaned_hparams[k] = float(v)
                    else:
                        cleaned_hparams[k] = v
                
                # ハイパーパラメータをWandBに送信
                wandb.config.update(cleaned_hparams)
                
                # メトリクスも記録
                cleaned_meters = {}
                for k, v in meters.items():
                    if isinstance(v, (Tensor, ndarray)):
                        cleaned_meters[k] = float(v)
                    else:
                        cleaned_meters[k] = v
                
                wandb.log(cleaned_meters)
                
            except Exception as e:
                logging.warning(f"Failed to log hparams to WandB: {e}")


def create_wandb_tensorboard_logger(logging_conf, use_wandb: bool = False):
    """
    WandB + TensorBoard統合ロガーを作成
    """
    return WandBTensorBoardLogger(logging_conf, use_wandb=use_wandb)