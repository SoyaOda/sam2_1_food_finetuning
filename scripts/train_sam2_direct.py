#!/usr/bin/env python3
"""
SAM2.1の学習を直接実行するスクリプト
Hydraの設定パス問題を回避するため、設定を直接読み込んで実行
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

# SAM2のパスを追加
sam2_path = Path(__file__).parent.parent / "external" / "sam2"
sys.path.insert(0, str(sam2_path))
os.chdir(sam2_path)

# 必要なモジュールをインポート
from training.utils.train_utils import makedir, register_omegaconf_resolvers
from iopath.common.file_io import g_pathmgr

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config_direct(config_path: str) -> DictConfig:
    """YAMLファイルを直接読み込んでOmegaConfオブジェクトに変換"""
    with open(config_path, 'r') as f:
        # @package _global_ ディレクティブを除去
        lines = f.readlines()
        if lines[0].startswith("# @package"):
            lines = lines[1:]
        yaml_str = ''.join(lines)
        config_dict = yaml.safe_load(yaml_str)
    
    # OmegaConfに変換
    cfg = OmegaConf.create(config_dict)
    return cfg


def single_proc_run(cfg):
    """単一GPUプロセスでの学習実行（簡易版）"""
    import torch.distributed as dist
    
    # 環境変数の設定
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0" 
    os.environ["WORLD_SIZE"] = "1"
    
    # OmegaConfリゾルバの登録
    register_omegaconf_resolvers()
    
    # 設定の解決
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg_resolved = OmegaConf.create(cfg_resolved)
    
    # トレーナーのインスタンス化と実行
    try:
        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()
    except Exception as e:
        logger.error(f"トレーナーの実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """メイン処理"""
    # 設定ファイルのパス
    config_path = "sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_test.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        sys.exit(1)
    
    logger.info(f"設定ファイルを読み込み中: {config_path}")
    
    # 設定を読み込み
    cfg = load_config_direct(config_path)
    
    # ログディレクトリの設定
    if not hasattr(cfg, 'launcher'):
        cfg.launcher = OmegaConf.create({})
    
    if not hasattr(cfg.launcher, 'experiment_log_dir'):
        cfg.launcher.experiment_log_dir = "./sam2_logs/direct_train"
    
    cfg.launcher.num_nodes = 1
    cfg.launcher.gpus_per_node = 1
    
    # ログディレクトリを作成
    makedir(cfg.launcher.experiment_log_dir)
    
    # 設定を保存
    config_save_path = os.path.join(cfg.launcher.experiment_log_dir, "config.yaml")
    with g_pathmgr.open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    logger.info(f"実験ログディレクトリ: {cfg.launcher.experiment_log_dir}")
    
    # 主要な設定項目を表示
    logger.info("=" * 60)
    logger.info("主要な設定項目:")
    logger.info(f"  データセット画像: {cfg.dataset.img_folder}")
    logger.info(f"  データセットアノテーション: {cfg.dataset.gt_folder}")
    logger.info(f"  学習リスト: {cfg.dataset.file_list_txt}")
    logger.info(f"  バッチサイズ: {cfg.scratch.train_batch_size}")
    logger.info(f"  エポック数: {cfg.scratch.num_epochs}")
    logger.info(f"  学習率: {cfg.scratch.base_lr}")
    logger.info("=" * 60)
    
    # GPU情報を表示
    if torch.cuda.is_available():
        logger.info(f"GPU使用: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("GPUが利用できません。CPUで実行します。")
    
    # 学習を開始
    logger.info("学習を開始します...")
    
    try:
        single_proc_run(cfg)
        logger.info("学習が正常に完了しました。")
    except KeyboardInterrupt:
        logger.info("学習が中断されました。")
    except Exception as e:
        logger.error(f"学習中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()