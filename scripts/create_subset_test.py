#!/usr/bin/env python3
"""
小規模サブセット（32枚）での過学習テスト用データセット作成・学習スクリプト
学習の基本動作確認と問題切り分けを目的とする
"""

import os
import sys
import random
import logging
import argparse
import shutil
from pathlib import Path

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('subset_test.log')
        ]
    )

def create_subset_dataset(source_list_file: str, subset_size: int = 32, output_dir: str = "data/subset_test"):
    """
    元データセットから小規模サブセットを作成
    
    Args:
        source_list_file: 元のファイルリスト（train.txt等）
        subset_size: サブセットのサイズ
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 元のファイルリストを読み込み
    with open(source_list_file, 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]
    
    logging.info(f"元データセットサイズ: {len(all_files)} ファイル")
    
    # ランダムサンプリング（再現性のためseed固定）
    random.seed(42)
    subset_files = random.sample(all_files, min(subset_size, len(all_files)))
    
    # サブセットファイルリストを保存
    subset_list_path = os.path.join(output_dir, "subset_train.txt")
    with open(subset_list_path, 'w') as f:
        for file_path in subset_files:
            f.write(f"{file_path}\n")
    
    logging.info(f"サブセット作成完了: {len(subset_files)} ファイル")
    logging.info(f"サブセットリスト: {subset_list_path}")
    
    # 検証用に同じサブセットを使用（過学習テストなので）
    val_list_path = os.path.join(output_dir, "subset_val.txt")
    shutil.copy(subset_list_path, val_list_path)
    
    return subset_list_path, val_list_path

def create_subset_config(base_config_path: str, subset_train_list: str, subset_val_list: str, output_dir: str):
    """
    サブセット用の設定ファイルを作成
    """
    config_output_path = os.path.join(output_dir, "sam2.1_hiera_b+_subset_test.yaml")
    
    # 元の設定を読み込み
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # サブセット用に修正
    modifications = [
        # エポック数を増やして過学習確認
        ('num_epochs: 40', 'num_epochs: 10'),
        # ログ頻度を上げる
        ('log_freq: 50', 'log_freq: 5'),
        # チェックポイント頻度を上げる
        ('save_freq: 200', 'save_freq: 25'),
        # 学習データリストを置換
        (f'file_list_txt: /home/soya/sam2_1_food_finetuning/data/foodmix_sa1b_splits/train.txt', 
         f'file_list_txt: {os.path.abspath(subset_train_list)}'),
        # 検証データリストを置換
        (f'file_list_txt: /home/soya/sam2_1_food_finetuning/data/foodmix_sa1b_splits/val.txt',
         f'file_list_txt: {os.path.abspath(subset_val_list)}'),
        # 実験ログディレクトリを変更
        ('experiment_log_dir: ./sam2_logs/foodmix_optimized',
         f'experiment_log_dir: {os.path.abspath(output_dir)}/logs'),
    ]
    
    for old, new in modifications:
        config_content = config_content.replace(old, new)
    
    # サブセット設定を保存
    with open(config_output_path, 'w') as f:
        f.write(config_content)
    
    logging.info(f"サブセット設定作成完了: {config_output_path}")
    return config_output_path

def run_subset_training(config_path: str):
    """
    サブセット学習を実行
    """
    # 相対パスでConfigを指定（sam2/configsからの相対）
    config_name = os.path.basename(config_path).replace('.yaml', '')
    
    # sam2ディレクトリに移動
    sam2_dir = os.path.abspath("external/sam2")
    original_dir = os.getcwd()
    
    try:
        os.chdir(sam2_dir)
        
        # 学習コマンドを構築
        training_script = os.path.join(original_dir, "scripts/train_sam2_memory_optimized.py")
        
        command = [
            sys.executable,
            training_script,
            "-c", f"sam2.1_training/{config_name}",
            "--use-cluster", "0",
            "--num-gpus", "1"
        ]
        
        logging.info("サブセット学習開始...")
        logging.info(f"コマンド: {' '.join(command)}")
        
        # 学習実行
        import subprocess
        process = subprocess.run(command, cwd=sam2_dir, capture_output=True, text=True)
        
        if process.returncode == 0:
            logging.info("サブセット学習完了")
            logging.info(f"標準出力: {process.stdout}")
        else:
            logging.error(f"サブセット学習失敗 (exit code: {process.returncode})")
            logging.error(f"標準エラー: {process.stderr}")
            
        return process.returncode == 0
        
    finally:
        os.chdir(original_dir)

def analyze_subset_results(log_dir: str):
    """
    サブセット学習結果を分析
    """
    log_path = os.path.join(log_dir, "logs")
    tensorboard_path = os.path.join(log_dir, "tensorboard")
    
    logging.info("=== サブセット学習結果分析 ===")
    
    # TensorBoardログの存在確認
    if os.path.exists(tensorboard_path):
        tb_files = list(Path(tensorboard_path).rglob("events.out.tfevents.*"))
        logging.info(f"TensorBoardログ: {len(tb_files)} ファイル")
        
        # 損失の傾向を確認（簡易版）
        if tb_files:
            logging.info("TensorBoard で損失の推移を確認してください:")
            logging.info(f"tensorboard --logdir {tensorboard_path}")
    else:
        logging.warning("TensorBoardログが見つかりません")
    
    # チェックポイントの確認
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    if os.path.exists(checkpoint_path):
        checkpoints = list(Path(checkpoint_path).glob("*.pt"))
        logging.info(f"チェックポイント: {len(checkpoints)} ファイル")
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            logging.info(f"最新チェックポイント: {latest_checkpoint}")
    else:
        logging.warning("チェックポイントが見つかりません")

def main():
    parser = argparse.ArgumentParser(description="SAM2.1 小規模サブセット過学習テスト")
    parser.add_argument("--subset-size", type=int, default=32, 
                       help="サブセットサイズ（デフォルト: 32）")
    parser.add_argument("--source-list", type=str, 
                       default="data/foodmix_sa1b_splits/train.txt",
                       help="元データのファイルリスト")
    parser.add_argument("--base-config", type=str,
                       default="external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_optimized.yaml",
                       help="ベース設定ファイル")
    parser.add_argument("--output-dir", type=str, default="data/subset_test",
                       help="出力ディレクトリ")
    parser.add_argument("--skip-training", action='store_true',
                       help="学習をスキップして分析のみ実行")
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("SAM2.1 小規模サブセット過学習テスト開始")
    logging.info("=" * 60)
    
    # サブセットデータセット作成
    if not args.skip_training:
        subset_train_list, subset_val_list = create_subset_dataset(
            args.source_list, 
            args.subset_size, 
            args.output_dir
        )
        
        # サブセット用設定作成
        config_path = create_subset_config(
            args.base_config,
            subset_train_list,
            subset_val_list,
            args.output_dir
        )
        
        # 設定をsam2/configsにコピー
        sam2_config_dir = "external/sam2/sam2/configs/sam2.1_training"
        config_name = os.path.basename(config_path)
        sam2_config_path = os.path.join(sam2_config_dir, config_name)
        shutil.copy(config_path, sam2_config_path)
        logging.info(f"設定をSAM2ディレクトリにコピー: {sam2_config_path}")
        
        # サブセット学習実行
        success = run_subset_training(sam2_config_path)
        
        if not success:
            logging.error("サブセット学習に失敗しました")
            return 1
    
    # 結果分析
    analyze_subset_results(os.path.join(args.output_dir, "logs"))
    
    logging.info("=" * 60)
    logging.info("サブセットテスト完了")
    logging.info("=" * 60)
    logging.info("次のステップ:")
    logging.info("1. TensorBoard でloss の推移を確認")
    logging.info("2. train_all_loss が下がっているか確認")
    logging.info("3. 可視化スクリプトで予測品質を確認")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())