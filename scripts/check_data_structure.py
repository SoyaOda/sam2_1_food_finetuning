#!/usr/bin/env python3
"""
データセット構造確認スクリプト
SA1BRawDataset用のデータが正しく準備されているか確認
"""

import os
import json
from pathlib import Path
import argparse


def check_data_structure(data_dir):
    """SA1BRawDataset用のデータ構造を確認"""
    data_path = Path(data_dir)
    
    print(f"データディレクトリ: {data_path}")
    print("=" * 60)
    
    # 必要なディレクトリの確認
    img_dir = data_path / "images"
    ann_dir = data_path / "annotations"
    
    # ディレクトリ存在確認
    if not img_dir.exists():
        print(f"❌ 画像ディレクトリが見つかりません: {img_dir}")
        return False
    print(f"✓ 画像ディレクトリ: {img_dir}")
    
    if not ann_dir.exists():
        print(f"❌ アノテーションディレクトリが見つかりません: {ann_dir}")
        return False
    print(f"✓ アノテーションディレクトリ: {ann_dir}")
    
    # ファイル数確認
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    ann_files = list(ann_dir.glob("*.json"))
    
    print(f"\n画像ファイル数: {len(img_files)}")
    print(f"アノテーションファイル数: {len(ann_files)}")
    
    # train.txt/val.txt確認
    train_txt = data_path / "train.txt"
    val_txt = data_path / "val.txt"
    
    if train_txt.exists():
        with open(train_txt) as f:
            train_count = sum(1 for _ in f)
        print(f"✓ train.txt: {train_count} ファイル")
    else:
        print(f"❌ train.txt が見つかりません")
    
    if val_txt.exists():
        with open(val_txt) as f:
            val_count = sum(1 for _ in f)
        print(f"✓ val.txt: {val_count} ファイル")
    else:
        print(f"❌ val.txt が見つかりません")
    
    # サンプルアノテーションの確認
    if ann_files:
        sample_ann = ann_files[0]
        print(f"\nサンプルアノテーション確認: {sample_ann.name}")
        try:
            with open(sample_ann) as f:
                data = json.load(f)
            
            # SA-1B形式の確認
            if "image" in data and "annotations" in data:
                print("✓ SA-1B形式のアノテーション")
                print(f"  - 画像情報: {data['image'].get('file_name', 'N/A')}")
                print(f"  - アノテーション数: {len(data['annotations'])}")
                
                if data['annotations']:
                    ann = data['annotations'][0]
                    if "segmentation" in ann:
                        if isinstance(ann["segmentation"], dict) and "counts" in ann["segmentation"]:
                            print("  ✓ RLE形式のセグメンテーション")
                        else:
                            print("  ❌ セグメンテーション形式が不正")
            else:
                print("❌ SA-1B形式ではありません")
        except Exception as e:
            print(f"❌ アノテーション読み込みエラー: {e}")
    
    # 画像とアノテーションの対応確認
    print("\n画像とアノテーションの対応確認（最初の5件）:")
    for i, img_file in enumerate(img_files[:5]):
        stem = img_file.stem
        ann_file = ann_dir / f"{stem}.json"
        if ann_file.exists():
            print(f"  ✓ {img_file.name} → {ann_file.name}")
        else:
            print(f"  ❌ {img_file.name} → アノテーションなし")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="SA1BRawDataset用データ構造確認")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/foodmix_sa1b",
        help="データディレクトリ"
    )
    
    args = parser.parse_args()
    
    if check_data_structure(args.data_dir):
        print("\n✅ データ構造の確認が完了しました")
    else:
        print("\n❌ データ構造に問題があります")
        print("\n修正方法:")
        print("1. python scripts/10_prepare_foodseg103.py")
        print("2. python scripts/11_prepare_uecfoodpix.py")
        print("3. python scripts/12_merge_to_sa1b.py")


if __name__ == "__main__":
    main()