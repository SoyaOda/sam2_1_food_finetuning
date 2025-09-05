#!/usr/bin/env python3
"""
学習レジューム機能付きスクリプト
中断された学習を自動的に再開
"""

import os
import sys
import argparse
import logging
import glob
from typing import Optional, Dict, Any
import torch

# SAM2のパスを追加
sam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "sam2"))
sys.path.insert(0, sam2_path)
sys.path.insert(0, os.path.dirname(__file__) + "/..")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """最新のチェックポイントファイルを見つける"""
    
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # チェックポイントファイルのパターン
    patterns = [
        "checkpoint.pt",
        "checkpoint_*.pt",
        "model_*.pt"
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
        all_checkpoints.extend(checkpoint_files)
    
    if not all_checkpoints:
        logging.warning(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    # 最新のファイルを選択（作成時刻順）
    latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
    
    logging.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def check_checkpoint_validity(checkpoint_path: str) -> bool:
    """チェックポイントファイルの有効性をチェック"""
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model', 'optimizer', 'epoch', 'steps']
        for key in required_keys:
            if key not in checkpoint:
                logging.error(f"Missing key '{key}' in checkpoint")
                return False
        
        logging.info(f"Checkpoint validation passed: {checkpoint_path}")
        logging.info(f"  Epoch: {checkpoint['epoch']}")
        logging.info(f"  Steps: {checkpoint['steps']}")
        
        if 'time_elapsed' in checkpoint:
            logging.info(f"  Time elapsed: {checkpoint['time_elapsed']:.2f}s")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return False


def get_experiment_info(experiment_dir: str) -> Dict[str, Any]:
    """実験ディレクトリから情報を取得"""
    
    info = {
        'experiment_dir': experiment_dir,
        'config_path': None,
        'checkpoint_dir': None,
        'log_dir': None,
        'tensorboard_dir': None
    }
    
    # 設定ファイル
    config_path = os.path.join(experiment_dir, 'config.yaml')
    if os.path.exists(config_path):
        info['config_path'] = config_path
    
    # チェックポイントディレクトリ
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        info['checkpoint_dir'] = checkpoint_dir
    
    # ログディレクトリ
    log_dir = os.path.join(experiment_dir, 'logs')
    if os.path.exists(log_dir):
        info['log_dir'] = log_dir
    
    # TensorBoardディレクトリ
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
    if os.path.exists(tensorboard_dir):
        info['tensorboard_dir'] = tensorboard_dir
    
    return info


def list_available_experiments(base_dir: str = "./sam2_logs") -> list:
    """利用可能な実験を一覧表示"""
    
    if not os.path.exists(base_dir):
        return []
    
    experiments = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            info = get_experiment_info(item_path)
            if info['checkpoint_dir']:  # チェックポイントがあるもののみ
                checkpoint_files = glob.glob(os.path.join(info['checkpoint_dir'], "*.pt"))
                if checkpoint_files:
                    experiments.append({
                        'name': item,
                        'path': item_path,
                        'info': info,
                        'checkpoints': len(checkpoint_files)
                    })
    
    return experiments


def interactive_experiment_selection() -> Optional[str]:
    """対話式で実験を選択"""
    
    experiments = list_available_experiments()
    
    if not experiments:
        print("No resumable experiments found.")
        return None
    
    print("\n" + "="*60)
    print("AVAILABLE EXPERIMENTS TO RESUME")
    print("="*60)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   Path: {exp['path']}")
        print(f"   Checkpoints: {exp['checkpoints']}")
        
        # 最新チェックポイントの情報
        latest_checkpoint = find_latest_checkpoint(exp['info']['checkpoint_dir'])
        if latest_checkpoint and check_checkpoint_validity(latest_checkpoint):
            try:
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                print(f"   Latest: Epoch {checkpoint['epoch']}, Step {checkpoint['steps']}")
            except:
                pass
        print()
    
    print("="*60)
    
    while True:
        try:
            choice = input(f"Select experiment (1-{len(experiments)}, or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                return experiments[idx]['path']
            else:
                print(f"Please enter a number between 1 and {len(experiments)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def main():
    parser = argparse.ArgumentParser(description="Resume SAM2.1 Training")
    parser.add_argument(
        "--experiment-dir", 
        type=str,
        help="Path to experiment directory to resume"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str,
        help="Specific checkpoint file to resume from"
    )
    parser.add_argument(
        "--list", 
        action='store_true',
        help="List available experiments to resume"
    )
    parser.add_argument(
        "--interactive", 
        action='store_true',
        help="Interactive experiment selection"
    )
    parser.add_argument("--memory-optimized", action='store_true',
                       help="Use memory-optimized trainer")
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s'
    )
    
    # 利用可能な実験をリスト表示
    if args.list:
        experiments = list_available_experiments()
        if experiments:
            print("\nAvailable experiments to resume:")
            for exp in experiments:
                print(f"  - {exp['name']} ({exp['checkpoints']} checkpoints)")
        else:
            print("No resumable experiments found.")
        return
    
    # 実験ディレクトリの決定
    experiment_dir = None
    
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
    elif args.interactive:
        experiment_dir = interactive_experiment_selection()
        if not experiment_dir:
            logging.info("No experiment selected. Exiting.")
            return
    else:
        # 最新の実験を自動選択
        experiments = list_available_experiments()
        if experiments:
            # 最新の実験（ディレクトリの更新時刻順）
            latest_exp = max(experiments, key=lambda x: os.path.getmtime(x['path']))
            experiment_dir = latest_exp['path']
            logging.info(f"Auto-selected latest experiment: {latest_exp['name']}")
        else:
            logging.error("No experiments found to resume. Use --interactive or --experiment-dir")
            return
    
    # 実験情報の取得
    exp_info = get_experiment_info(experiment_dir)
    
    if not exp_info['checkpoint_dir']:
        logging.error(f"No checkpoint directory found in {experiment_dir}")
        return
    
    # チェックポイントの確認
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(exp_info['checkpoint_dir'])
    
    if not checkpoint_path or not check_checkpoint_validity(checkpoint_path):
        logging.error("No valid checkpoint found")
        return
    
    # 設定ファイルの確認
    if not exp_info['config_path']:
        logging.error(f"No config file found in {experiment_dir}")
        return
    
    logging.info("="*60)
    logging.info("RESUMING TRAINING")
    logging.info("="*60)
    logging.info(f"Experiment: {os.path.basename(experiment_dir)}")
    logging.info(f"Config: {exp_info['config_path']}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info("="*60)
    
    # 学習スクリプトの選択と実行
    if args.memory_optimized:
        script_path = "scripts/train_sam2_memory_optimized.py"
    else:
        script_path = "scripts/train_sam2_wrapper.py"
    
    # 設定名を抽出（config.yamlから）
    try:
        with open(exp_info['config_path'], 'r') as f:
            import yaml
            config_data = yaml.safe_load(f)
        
        # オリジナルの設定名を推定（実験ディレクトリ名から）
        exp_name = os.path.basename(experiment_dir)
        if exp_name.startswith("sam2.1_training_"):
            config_name = exp_name.replace("sam2.1_training_", "sam2.1_training/")
        else:
            # デフォルト設定を使用
            config_name = "sam2.1_training/sam2.1_hiera_b+_foodmix_optimized"
        
    except Exception as e:
        logging.warning(f"Could not parse config file: {e}")
        config_name = "sam2.1_training/sam2.1_hiera_b+_foodmix_optimized"
    
    # 学習を実行
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    
    cmd_parts = [
        "python", script_path,
        "-c", config_name,
        "--use-cluster", "0",
        "--num-gpus", "1"
    ]
    
    if args.memory_optimized:
        cmd_parts.extend(["--memory-check-interval", "50"])
    
    cmd = " ".join(cmd_parts)
    
    logging.info(f"Executing: {cmd}")
    
    # 環境変数でチェックポイントパスを設定
    os.environ['RESUME_CHECKPOINT'] = checkpoint_path
    
    # 実行
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        logging.info("Training completed successfully")
    else:
        logging.error(f"Training failed with exit code: {exit_code}")


if __name__ == "__main__":
    main()