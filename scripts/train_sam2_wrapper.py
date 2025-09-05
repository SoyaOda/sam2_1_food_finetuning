#!/usr/bin/env python3
"""
SAM2.1学習のラッパースクリプト
Hydraの設定パス問題を回避するため、initialize_config_dirを使用
"""

import os
import sys
import argparse

# SAM2のパスを追加
sam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "sam2"))
sys.path.insert(0, sam2_path)
os.chdir(sam2_path)

# Hydraの初期化を上書き
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from training.utils.train_utils import register_omegaconf_resolvers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", 
        required=True,
        help="Config name (e.g., sam2.1_training/sam2.1_hiera_b+_foodmix_simple)"
    )
    parser.add_argument("--use-cluster", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    args = parser.parse_args()
    
    # Hydraをリセット
    GlobalHydra.instance().clear()
    
    # カスタム初期化
    config_dir = os.path.abspath("sam2/configs")
    initialize_config_dir(config_dir=config_dir, version_base="1.2")
    
    # OmegaConfリゾルバの登録
    register_omegaconf_resolvers()
    
    # 設定を読み込み
    cfg = compose(config_name=args.config)
    
    # argsを設定に追加
    args.use_cluster = bool(args.use_cluster)
    
    # train.pyのmain関数を呼び出し
    from training.train import main as train_main
    
    # 簡易的なargsオブジェクトを作成
    class Args:
        def __init__(self, config, use_cluster, num_gpus, num_nodes):
            self.config = config
            self.use_cluster = use_cluster
            self.num_gpus = num_gpus
            self.num_nodes = num_nodes
            self.partition = None
            self.account = None
            self.qos = None
    
    train_args = Args(args.config, args.use_cluster, args.num_gpus, args.num_nodes)
    
    # mainを直接呼び出すのではなく、single_node_runnerを使用
    from training.train import single_node_runner
    import random
    
    # ログディレクトリの設定
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config.replace("/", "_")
        )
    
    # ディレクトリ作成
    os.makedirs(cfg.launcher.experiment_log_dir, exist_ok=True)
    
    # 設定を保存
    from omegaconf import OmegaConf
    with open(os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w") as f:
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
    
    # 実行
    cfg.launcher.num_nodes = 1
    cfg.launcher.gpus_per_node = args.num_gpus
    
    print("=" * 60)
    print("SAM2.1 Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Experiment dir: {cfg.launcher.experiment_log_dir}")
    print(f"Dataset images: {cfg.dataset.img_folder}")
    print(f"Dataset annotations: {cfg.dataset.gt_folder}")
    print(f"Training list: {cfg.dataset.file_list_txt}")
    print(f"Batch size: {cfg.scratch.train_batch_size}")
    print(f"Epochs: {cfg.scratch.num_epochs}")
    print(f"Learning rate: {cfg.scratch.base_lr}")
    print("=" * 60)
    
    single_node_runner(cfg, main_port)

if __name__ == "__main__":
    main()