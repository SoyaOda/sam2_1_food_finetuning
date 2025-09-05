#!/usr/bin/env python
"""
FoodSeg103データセットをHugging Faceからダウンロード
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

def download_and_save_foodseg103():
    """FoodSeg103をHugging Faceからダウンロードしてローカルに保存"""
    
    print("FoodSeg103データセットをHugging Faceからダウンロード中...")
    print("注意: このデータセットは研究目的でのみ使用可能です")
    
    # データセットをロード
    try:
        dataset = load_dataset("EduardoPacheco/FoodSeg103")
    except Exception as e:
        print(f"エラー: データセットのダウンロードに失敗しました: {e}")
        print("インターネット接続を確認してください")
        return False
    
    print(f"データセット構造:")
    print(dataset)
    
    # 保存先ディレクトリを作成
    base_dir = Path("data/FoodSeg103")
    img_dir = base_dir / "Images"
    mask_dir = base_dir / "Masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # trainとvalidationを分けて保存
    for split in ["train", "validation"]:
        split_data = dataset[split]
        split_img_dir = img_dir / split
        split_mask_dir = mask_dir / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{split}データを保存中...")
        for idx in tqdm(range(len(split_data))):
            sample = split_data[idx]
            
            # 画像とマスクを取得
            image = sample["image"]  # PIL Image
            mask = sample["label"]   # PIL Image
            
            # ファイル名を生成
            filename_base = f"{split}_{idx:05d}"
            img_path = split_img_dir / f"{filename_base}.jpg"
            mask_path = split_mask_dir / f"{filename_base}.png"
            
            # 保存
            image.save(img_path)
            mask.save(mask_path)
    
    print(f"\nダウンロード完了!")
    print(f"保存先: {base_dir}")
    
    # データセット情報を保存
    info_file = base_dir / "dataset_info.txt"
    with open(info_file, "w") as f:
        f.write("FoodSeg103 Dataset\n")
        f.write("==================\n")
        f.write(f"Train samples: {len(dataset['train'])}\n")
        f.write(f"Validation samples: {len(dataset['validation'])}\n")
        f.write("\nDirectory structure:\n")
        f.write("- Images/\n")
        f.write("  - train/\n")
        f.write("  - test/\n")
        f.write("- Masks/\n")
        f.write("  - train/\n")
        f.write("  - test/\n")
        f.write("\nNote: Downloaded from Hugging Face (EduardoPacheco/FoodSeg103)\n")
    
    return True

if __name__ == "__main__":
    # datasetsライブラリのインストール確認
    try:
        import datasets
    except ImportError:
        print("datasetsライブラリがインストールされていません")
        print("インストール中...")
        os.system("pip install datasets --break-system-packages")
        import datasets
    
    success = download_and_save_foodseg103()
    sys.exit(0 if success else 1)