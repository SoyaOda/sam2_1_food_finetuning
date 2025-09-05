#!/usr/bin/env python3
"""
12_merge_to_sa1b.py - FoodSeg103とUEC-FoodPixのデータを統合し、train/valスプリットを作成

両データセットの処理済みデータ（SA-1B形式）を統合し、
学習用と検証用のファイルリストを生成する。
"""

import os
import json
import random
import glob
import argparse
from pathlib import Path
from typing import List, Tuple


def check_annotation_integrity(data_dir: Path) -> Tuple[List[str], List[str]]:
    """
    画像とアノテーションの対応をチェックし、有効なペアのリストを返す
    
    Returns:
        valid_pairs: 有効な画像ファイル名のリスト
        missing_annotations: アノテーションが見つからない画像のリスト
    """
    img_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    
    if not img_dir.exists() or not ann_dir.exists():
        print(f"警告: {data_dir} に必要なディレクトリが見つかりません")
        return [], []
    
    # 画像ファイルの一覧取得
    img_files = sorted(
        list(img_dir.glob("*.jpg")) + 
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )
    
    valid_pairs = []
    missing_annotations = []
    
    for img_path in img_files:
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.json"
        
        if ann_path.exists():
            # JSONの妥当性チェック
            try:
                with open(ann_path, "r") as f:
                    data = json.load(f)
                    if "image" in data and "annotations" in data:
                        valid_pairs.append(img_path.name)
                    else:
                        print(f"警告: {ann_path.name} のJSON形式が不正です")
            except json.JSONDecodeError:
                print(f"警告: {ann_path.name} のJSONパースに失敗しました")
        else:
            missing_annotations.append(img_path.name)
    
    return valid_pairs, missing_annotations


def create_train_val_split(
    valid_pairs: List[str], 
    split_ratio: float = 0.9,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    データをtrain/valに分割
    
    Args:
        valid_pairs: 有効な画像ファイル名のリスト
        split_ratio: 訓練データの割合（0.9 = 90%が訓練用）
        random_seed: ランダムシード
    
    Returns:
        train_list: 訓練用ファイルリスト
        val_list: 検証用ファイルリスト
    """
    random.seed(random_seed)
    
    # シャッフル
    shuffled = valid_pairs.copy()
    random.shuffle(shuffled)
    
    # 分割
    n_total = len(shuffled)
    n_train = int(n_total * split_ratio)
    
    train_list = shuffled[:n_train]
    val_list = shuffled[n_train:]
    
    return train_list, val_list


def write_file_list(file_path: Path, file_list: List[str]):
    """
    ファイルリストをテキストファイルに書き込み
    """
    with open(file_path, "w") as f:
        for fname in file_list:
            f.write(f"{fname}\n")


def merge_datasets(
    data_dir: str,
    split_ratio: float = 0.9,
    random_seed: int = 42,
    separate_datasets: bool = False
):
    """
    FoodSeg103とUEC-FoodPixのデータを統合
    
    Args:
        data_dir: foodmix_sa1bディレクトリのパス
        split_ratio: 訓練データの割合
        random_seed: ランダムシード
        separate_datasets: データセット別に分割を作成するか
    """
    data_path = Path(data_dir)
    
    print("=== データセット統合処理を開始 ===")
    print(f"データディレクトリ: {data_path}")
    
    # データ整合性チェック
    valid_pairs, missing = check_annotation_integrity(data_path)
    
    if not valid_pairs:
        print("エラー: 有効なデータペアが見つかりません")
        print("前処理スクリプト（10_prepare_foodseg103.py, 11_prepare_uecfoodpix.py）を実行してください")
        return
    
    print(f"\n有効なデータペア数: {len(valid_pairs)}")
    if missing:
        print(f"アノテーションが見つからない画像数: {len(missing)}")
        if len(missing) <= 10:
            for m in missing:
                print(f"  - {m}")
        else:
            print(f"  （最初の10件）")
            for m in missing[:10]:
                print(f"  - {m}")
    
    # データセット別に分類（オプション）
    if separate_datasets:
        foodseg_pairs = [p for p in valid_pairs if not p.startswith("uec_")]
        uec_pairs = [p for p in valid_pairs if p.startswith("uec_")]
        
        print(f"\nFoodSeg103: {len(foodseg_pairs)} ペア")
        print(f"UEC-FoodPix: {len(uec_pairs)} ペア")
        
        # 各データセットで個別にスプリット
        if foodseg_pairs:
            fs_train, fs_val = create_train_val_split(foodseg_pairs, split_ratio, random_seed)
            print(f"  FoodSeg103 - Train: {len(fs_train)}, Val: {len(fs_val)}")
        else:
            fs_train, fs_val = [], []
        
        if uec_pairs:
            uec_train, uec_val = create_train_val_split(uec_pairs, split_ratio, random_seed + 1)
            print(f"  UEC-FoodPix - Train: {len(uec_train)}, Val: {len(uec_val)}")
        else:
            uec_train, uec_val = [], []
        
        # 統合
        train_list = fs_train + uec_train
        val_list = fs_val + uec_val
        
        # 統合後にシャッフル
        random.shuffle(train_list)
        random.shuffle(val_list)
    else:
        # 全データを混合してスプリット
        train_list, val_list = create_train_val_split(valid_pairs, split_ratio, random_seed)
    
    print(f"\n=== 最終スプリット ===")
    print(f"Train: {len(train_list)} ファイル")
    print(f"Val: {len(val_list)} ファイル")
    print(f"合計: {len(train_list) + len(val_list)} ファイル")
    
    # ファイルリスト書き込み
    train_file = data_path / "train.txt"
    val_file = data_path / "val.txt"
    
    write_file_list(train_file, train_list)
    write_file_list(val_file, val_list)
    
    print(f"\n=== ファイルリスト生成完了 ===")
    print(f"Train list: {train_file}")
    print(f"Val list: {val_file}")
    
    # サンプル表示
    print(f"\nTrainデータのサンプル（最初の5件）:")
    for i, fname in enumerate(train_list[:5]):
        print(f"  {i+1}. {fname}")
    
    print(f"\nValデータのサンプル（最初の5件）:")
    for i, fname in enumerate(val_list[:5]):
        print(f"  {i+1}. {fname}")
    
    # 統計情報
    print(f"\n=== データセット統計 ===")
    
    # データセット別の統計
    foodseg_count = sum(1 for p in valid_pairs if not p.startswith("uec_"))
    uec_count = sum(1 for p in valid_pairs if p.startswith("uec_"))
    
    if foodseg_count > 0:
        print(f"FoodSeg103の割合: {foodseg_count}/{len(valid_pairs)} ({100*foodseg_count/len(valid_pairs):.1f}%)")
    if uec_count > 0:
        print(f"UEC-FoodPixの割合: {uec_count}/{len(valid_pairs)} ({100*uec_count/len(valid_pairs):.1f}%)")
    
    # アノテーション統計の簡易チェック
    ann_dir = data_path / "annotations"
    total_instances = 0
    sample_count = min(100, len(valid_pairs))  # 最大100ファイルをサンプリング
    
    for fname in random.sample(valid_pairs, sample_count):
        stem = Path(fname).stem
        ann_path = ann_dir / f"{stem}.json"
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
                total_instances += len(data.get("annotations", []))
        except:
            pass
    
    if sample_count > 0:
        avg_instances = total_instances / sample_count
        print(f"\n平均インスタンス数（{sample_count}ファイルのサンプル）: {avg_instances:.1f}")
        estimated_total = int(avg_instances * len(valid_pairs))
        print(f"推定総インスタンス数: 約{estimated_total:,}")


def main():
    parser = argparse.ArgumentParser(
        description="FoodSeg103とUEC-FoodPixのデータを統合し、train/valスプリットを作成"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/foodmix_sa1b",
        help="SA-1B形式のデータが格納されているディレクトリ"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        help="訓練データの割合（デフォルト: 0.9）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード（デフォルト: 42）"
    )
    parser.add_argument(
        "--separate-datasets",
        action="store_true",
        help="データセット別に分割を作成してから統合"
    )
    
    args = parser.parse_args()
    
    # 処理実行
    merge_datasets(
        args.data_dir,
        args.split_ratio,
        args.seed,
        args.separate_datasets
    )
    
    print("\n=== 処理完了 ===")
    print("次のステップ:")
    print("1. python scripts/13_optional_resize_1024.py - （任意）画像の1024x1024リサイズ")
    print("2. python scripts/20_train_food_sam2.py - SAM2.1の学習開始")
    print("\nまたは直接学習を開始:")
    print("cd external/sam2")
    print("python training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml")


if __name__ == "__main__":
    main()