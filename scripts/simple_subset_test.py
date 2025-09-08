#!/usr/bin/env python3
"""
シンプルなサブセットテスト - 基本的な学習動作確認用
"""

import os
import sys
import random
import logging
import shutil

def create_simple_subset():
    """シンプルなサブセット作成"""
    
    # 元のデータファイルリストを読み込み
    train_list_path = "data/foodmix_sa1b_splits/train.txt"
    if not os.path.exists(train_list_path):
        logging.error(f"訓練データリストが見つかりません: {train_list_path}")
        return False
    
    with open(train_list_path, 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]
    
    # 8枚をランダムサンプリング
    random.seed(42)
    subset_files = random.sample(all_files, 8)
    
    # サブセット出力ディレクトリ作成
    subset_dir = "data/simple_subset"
    os.makedirs(subset_dir, exist_ok=True)
    
    # サブセットファイル保存
    subset_train_path = os.path.join(subset_dir, "train.txt")
    with open(subset_train_path, 'w') as f:
        for file_id in subset_files:
            # 拡張子を除去
            if file_id.endswith('.jpg'):
                file_id = file_id[:-4]
            f.write(f"{file_id}\n")
    
    # 検証用も同じファイルを使用
    subset_val_path = os.path.join(subset_dir, "val.txt")
    shutil.copy(subset_train_path, subset_val_path)
    
    logging.info(f"サブセット作成完了: {len(subset_files)} ファイル")
    logging.info(f"サブセットファイル: {subset_train_path}")
    
    return subset_train_path, subset_val_path

def create_simple_config():
    """シンプルなサブセット用設定作成"""
    
    base_config_path = "external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_optimized.yaml"
    
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # 最小限の修正
    modifications = [
        # エポック数を削減
        ('num_epochs: 40', 'num_epochs: 3'),
        # チェックポイント頻度
        ('save_freq: 200', 'save_freq: 10'),
        # 実験ディレクトリ
        ('experiment_log_dir: ./sam2_logs/foodmix_optimized',
         'experiment_log_dir: ./data/simple_subset/logs'),
        # 訓練データファイルリスト
        ('file_list_txt: /home/soya/sam2_1_food_finetuning/data/foodmix_sa1b_splits/train.txt',
         'file_list_txt: /home/soya/sam2_1_food_finetuning/data/simple_subset/train.txt'),
        # 検証データファイルリスト
        ('file_list_txt: /home/soya/sam2_1_food_finetuning/data/foodmix_sa1b_splits/val.txt',
         'file_list_txt: /home/soya/sam2_1_food_finetuning/data/simple_subset/val.txt'),
    ]
    
    for old, new in modifications:
        config_content = config_content.replace(old, new)
    
    # サブセット設定保存
    subset_config_path = "external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_simple_subset.yaml"
    with open(subset_config_path, 'w') as f:
        f.write(config_content)
    
    logging.info(f"サブセット設定作成: {subset_config_path}")
    return subset_config_path

def main():
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s [%(levelname)s] %(message)s')
    
    logging.info("=== シンプルサブセットテスト開始 ===")
    
    # サブセット作成
    subset_result = create_simple_subset()
    if not subset_result:
        return 1
    
    # 設定作成
    config_path = create_simple_config()
    
    logging.info("=== 準備完了 ===")
    logging.info("次のコマンドで学習を開始:")
    logging.info("cd external/sam2 && python ../../scripts/train_sam2_memory_optimized.py \\")
    logging.info("  -c sam2.1_training/sam2.1_hiera_b+_simple_subset --use-cluster 0 --num-gpus 1")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())