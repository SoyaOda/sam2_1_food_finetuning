#!/usr/bin/env python3
"""
データセット統合・スプリット作成スクリプト
UEC-FoodPix Complete + FoodSeg103 を統合してSAM2.1学習用のスプリットを作成

学習スクリプト対応:
bash scripts/train_with_monitoring.sh --memory-optimized --auto-resume
"""

import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict
import numpy as np

def verify_data_integrity(data_dir: Path) -> Dict[str, int]:
    """データの整合性を確認"""
    img_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    
    img_files = set(f.stem for f in img_dir.glob("*.jpg"))
    ann_files = set(f.stem for f in ann_dir.glob("*.json"))
    
    # ペアの一致確認
    matching_pairs = img_files & ann_files
    
    # データセット別の統計
    stats = {
        "total_images": len(img_files),
        "total_annotations": len(ann_files),
        "matching_pairs": len(matching_pairs),
        "uec_train": len([f for f in img_files if f.startswith("uec_train_")]),
        "uec_test": len([f for f in img_files if f.startswith("uec_test_")]),
        "foodseg_train": len([f for f in img_files if f.startswith("train_train_")]),
        "foodseg_val": len([f for f in img_files if f.startswith("val_train_")])
    }
    
    return stats

def load_annotation_metadata(ann_file: Path) -> Dict:
    """アノテーションファイルからメタデータを読み込み"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    return {
        "num_annotations": len(data.get("annotations", [])),
        "image_width": data.get("image", {}).get("width", 0),
        "image_height": data.get("image", {}).get("height", 0),
        "total_area": sum(ann.get("area", 0) for ann in data.get("annotations", []))
    }

def create_balanced_splits(
    all_files: List[str], 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """バランスの取れたデータセットスプリットを作成"""
    
    # データセット別にグループ化
    uec_train = [f for f in all_files if f.startswith("uec_train_")]
    uec_test = [f for f in all_files if f.startswith("uec_test_")]
    foodseg_train = [f for f in all_files if f.startswith("train_train_")]
    foodseg_val = [f for f in all_files if f.startswith("val_train_")]
    
    random.seed(seed)
    
    # 各グループをシャッフル
    random.shuffle(uec_train)
    random.shuffle(uec_test)
    random.shuffle(foodseg_train)
    random.shuffle(foodseg_val)
    
    # UECデータを分割（元々train/testに分かれているが、さらに細分化）
    uec_train_count = int(len(uec_train) * train_ratio)
    uec_val_count = int(len(uec_train) * val_ratio)
    
    uec_test_count = int(len(uec_test) * train_ratio)
    uec_test_val_count = int(len(uec_test) * val_ratio)
    
    # FoodSeg103データを分割
    fs_train_count = int(len(foodseg_train) * train_ratio)
    fs_val_count = int(len(foodseg_train) * val_ratio)
    
    fs_val_train_count = int(len(foodseg_val) * 0.5)  # FoodSeg val の半分を train に
    
    # 最終的な分割
    final_train = (
        uec_train[:uec_train_count] +
        uec_test[:uec_test_count] +
        foodseg_train[:fs_train_count] +
        foodseg_val[:fs_val_train_count]
    )
    
    final_val = (
        uec_train[uec_train_count:uec_train_count + uec_val_count] +
        uec_test[uec_test_count:uec_test_count + uec_test_val_count] +
        foodseg_train[fs_train_count:fs_train_count + fs_val_count] +
        foodseg_val[fs_val_train_count:]
    )
    
    final_test = (
        uec_train[uec_train_count + uec_val_count:] +
        uec_test[uec_test_count + uec_test_val_count:]
    )
    
    # 残りをtestに追加（FoodSeg103のtrainの残り）
    final_test.extend(foodseg_train[fs_train_count + fs_val_count:])
    
    return final_train, final_val, final_test

def create_sam2_compatible_splits(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15,
    copy_files: bool = False,
    seed: int = 42
):
    """SAM2.1学習互換のスプリットを作成"""
    
    print("📊 データ整合性チェック中...")
    stats = verify_data_integrity(data_dir)
    
    print(f"📈 データセット統計:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    if stats["matching_pairs"] != stats["total_images"]:
        print("⚠️  警告: 画像とアノテーションのペア不一致があります")
    
    # 全ファイルリストを取得（マッチングペアのみ）
    img_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    
    img_files = set(f.stem for f in img_dir.glob("*.jpg"))
    ann_files = set(f.stem for f in ann_dir.glob("*.json"))
    matching_files = list(img_files & ann_files)
    
    print(f"\n🔄 スプリット作成中 (train:{train_ratio}, val:{val_ratio}, test:{test_ratio})...")
    train_files, val_files, test_files = create_balanced_splits(
        matching_files, train_ratio, val_ratio, test_ratio, seed
    )
    
    print(f"📋 スプリット結果:")
    print(f"  Train: {len(train_files):,} ファイル")
    print(f"  Val:   {len(val_files):,} ファイル")
    print(f"  Test:  {len(test_files):,} ファイル")
    print(f"  Total: {len(train_files) + len(val_files) + len(test_files):,} ファイル")
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SAM2.1学習用のファイルリスト作成
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    for split_name, file_list in splits.items():
        # .txtファイル（画像ファイル名）
        txt_file = output_dir / f"{split_name}.txt"
        with open(txt_file, 'w') as f:
            for file_stem in sorted(file_list):
                f.write(f"{file_stem}.jpg\n")
        
        print(f"📄 {split_name}.txt 作成: {len(file_list):,} エントリ")
        
        # オプション: ファイルをコピー
        if copy_files:
            split_img_dir = output_dir / f"{split_name}_images"
            split_ann_dir = output_dir / f"{split_name}_annotations"
            split_img_dir.mkdir(exist_ok=True)
            split_ann_dir.mkdir(exist_ok=True)
            
            print(f"📂 {split_name} ファイルコピー中...")
            for file_stem in file_list:
                # 画像をコピー
                src_img = img_dir / f"{file_stem}.jpg"
                dst_img = split_img_dir / f"{file_stem}.jpg"
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                
                # アノテーションをコピー
                src_ann = ann_dir / f"{file_stem}.json"
                dst_ann = split_ann_dir / f"{file_stem}.json"
                if src_ann.exists():
                    shutil.copy2(src_ann, dst_ann)
    
    # データセット情報をJSONで保存
    dataset_info = {
        "total_samples": len(matching_files),
        "splits": {
            "train": {"count": len(train_files), "ratio": len(train_files) / len(matching_files)},
            "val": {"count": len(val_files), "ratio": len(val_files) / len(matching_files)},
            "test": {"count": len(test_files), "ratio": len(test_files) / len(matching_files)}
        },
        "dataset_composition": {
            "uec_train_samples": len([f for f in matching_files if f.startswith("uec_train_")]),
            "uec_test_samples": len([f for f in matching_files if f.startswith("uec_test_")]),
            "foodseg_train_samples": len([f for f in matching_files if f.startswith("train_train_")]),
            "foodseg_val_samples": len([f for f in matching_files if f.startswith("val_train_")])
        },
        "paths": {
            "source_images": str(img_dir),
            "source_annotations": str(ann_dir),
            "output_directory": str(output_dir)
        },
        "sam2_config": {
            "img_folder": str(data_dir / "images"),
            "gt_folder": str(data_dir / "annotations"),
            "train_filelist": str(output_dir / "train.txt"),
            "val_filelist": str(output_dir / "val.txt"),
            "test_filelist": str(output_dir / "test.txt")
        }
    }
    
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"ℹ️  データセット情報: {info_file}")
    
    return dataset_info

def analyze_dataset_distribution(data_dir: Path, splits_dir: Path):
    """データセット分布の詳細分析（簡易版）"""
    print("\n📊 データセット分布分析中...")
    
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            continue
            
        with open(split_file, 'r') as f:
            files = [line.strip().replace('.jpg', '') for line in f if line.strip()]
        
        # データセット構成分析
        uec_count = len([f for f in files if f.startswith("uec_")])
        foodseg_count = len([f for f in files if f.startswith(("train_train_", "val_train_"))])
        
        print(f"  {split_name.upper()}: {len(files):,} ファイル")
        print(f"    UEC-FoodPix: {uec_count:,} ファイル")
        print(f"    FoodSeg103: {foodseg_count:,} ファイル")

def main():
    parser = argparse.ArgumentParser(
        description="データセット統合・スプリット作成（SAM2.1学習用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な統合とスプリット作成
  python scripts/12_merge_to_sa1b.py --input data/foodmix_sa1b --output data/foodmix_sa1b_splits
  
  # カスタム比率でスプリット
  python scripts/12_merge_to_sa1b.py --input data/foodmix_sa1b --output data/foodmix_sa1b_splits \\
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
    
  # ファイルコピー付きで実行
  python scripts/12_merge_to_sa1b.py --input data/foodmix_sa1b --output data/foodmix_sa1b_splits --copy-files

学習実行:
  bash scripts/train_with_monitoring.sh --memory-optimized --auto-resume
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/foodmix_sa1b",
        help="前処理済みデータのディレクトリ"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/foodmix_sa1b_splits",
        help="統合スプリットの出力ディレクトリ"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="学習用データの比率 (default: 0.7)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="検証用データの比率 (default: 0.15)"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="テスト用データの比率 (default: 0.15)"
    )
    
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="ファイルを実際にコピーする（容量注意）"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード (default: 42)"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="分析のみ実行（ファイル作成なし）"
    )
    
    args = parser.parse_args()
    
    # 比率の合計チェック
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ エラー: 比率の合計が1.0ではありません ({total_ratio})")
        return
    
    data_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not data_dir.exists():
        print(f"❌ エラー: 入力ディレクトリが存在しません: {data_dir}")
        return
    
    print("🚀 データセット統合・スプリット作成開始")
    print(f"入力: {data_dir}")
    print(f"出力: {output_dir}")
    
    if args.analyze_only:
        verify_data_integrity(data_dir)
        return
    
    # スプリット作成実行
    dataset_info = create_sam2_compatible_splits(
        data_dir=data_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_files=args.copy_files,
        seed=args.seed
    )
    
    # 分布分析
    analyze_dataset_distribution(data_dir, output_dir)
    
    print(f"\\n✅ 統合・スプリット作成完了!")
    print(f"\\n📋 SAM2.1学習での使用方法:")
    print(f"   img_folder: {dataset_info['sam2_config']['img_folder']}")
    print(f"   gt_folder: {dataset_info['sam2_config']['gt_folder']}")
    print(f"   train_filelist: {dataset_info['sam2_config']['train_filelist']}")
    print(f"\\n🎯 学習実行:")
    print(f"   bash scripts/train_with_monitoring.sh --memory-optimized --auto-resume")

if __name__ == "__main__":
    main()