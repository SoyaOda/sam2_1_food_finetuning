#!/usr/bin/env python
"""
SA-1B形式に変換されたデータを検証するスクリプト
- JSONフォーマットの妥当性チェック
- RLEマスクのデコード確認
- 画像とアノテーションの対応確認
- データの統計情報表示
"""

import os
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
from tqdm import tqdm

def decode_rle(rle_dict, height, width):
    """RLEをバイナリマスクにデコード"""
    if isinstance(rle_dict["counts"], str):
        rle_dict = rle_dict.copy()
        rle_dict["counts"] = rle_dict["counts"].encode("ascii")
    
    binary_mask = mask_utils.decode(rle_dict)
    return binary_mask.reshape((height, width), order="F")

def verify_json_structure(json_path):
    """JSONファイルの構造を検証"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    errors = []
    
    # 必須フィールドのチェック
    if "image" not in data:
        errors.append("'image'フィールドが存在しません")
    else:
        img_info = data["image"]
        required_img_fields = ["width", "height", "file_name"]
        for field in required_img_fields:
            if field not in img_info:
                errors.append(f"'image.{field}'フィールドが存在しません")
    
    if "annotations" not in data:
        errors.append("'annotations'フィールドが存在しません")
    else:
        anns = data["annotations"]
        if not isinstance(anns, list):
            errors.append("'annotations'はリストである必要があります")
        else:
            for i, ann in enumerate(anns):
                required_ann_fields = ["id", "segmentation", "bbox", "area"]
                for field in required_ann_fields:
                    if field not in ann:
                        errors.append(f"annotations[{i}]に'{field}'フィールドが存在しません")
                        break
                
                # RLEフォーマットのチェック
                if "segmentation" in ann:
                    seg = ann["segmentation"]
                    if not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
                        errors.append(f"annotations[{i}]のsegmentationがRLE形式ではありません")
    
    return errors, data

def visualize_sample(img_dir, ann_dir, sample_name=None):
    """サンプルデータの可視化"""
    # ランダムに選択するか指定されたサンプルを使用
    if sample_name is None:
        ann_files = list(Path(ann_dir).glob("*.json"))
        if not ann_files:
            print("アノテーションファイルが見つかりません")
            return
        ann_path = random.choice(ann_files)
        sample_name = ann_path.stem
    else:
        ann_path = Path(ann_dir) / f"{sample_name}.json"
    
    # 対応する画像を探す
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = Path(img_dir) / f"{sample_name}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    
    if img_path is None:
        print(f"画像ファイルが見つかりません: {sample_name}")
        return
    
    # データを読み込み
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    
    errors, data = verify_json_structure(ann_path)
    if errors:
        print(f"JSONエラー: {errors}")
        return
    
    height, width = data["image"]["height"], data["image"]["width"]
    anns = data["annotations"]
    
    # マスクをデコードして可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元画像
    axes[0].imshow(img_array)
    axes[0].set_title(f"Original Image\n{sample_name}")
    axes[0].axis('off')
    
    # 全インスタンスマスク
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for i, ann in enumerate(anns):
        mask = decode_rle(ann["segmentation"], height, width)
        combined_mask[mask > 0] = (i % 20) + 1
    
    axes[1].imshow(combined_mask, cmap='tab20')
    axes[1].set_title(f"Instance Masks\n({len(anns)} instances)")
    axes[1].axis('off')
    
    # オーバーレイ
    overlay = img_array.copy().astype(float)
    mask_overlay = np.zeros_like(overlay)
    
    for i, ann in enumerate(anns):
        mask = decode_rle(ann["segmentation"], height, width)
        color = colors[i % 20][:3]
        for c in range(3):
            mask_overlay[:, :, c][mask > 0] = color[c] * 255
    
    overlay = overlay * 0.5 + mask_overlay * 0.5
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"verification_{sample_name}.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\n{sample_name}の統計:")
    print(f"  画像サイズ: {width} x {height}")
    print(f"  インスタンス数: {len(anns)}")
    print(f"  平均面積: {np.mean([ann['area'] for ann in anns]):.0f} pixels")
    print(f"  最小面積: {min([ann['area'] for ann in anns])} pixels")
    print(f"  最大面積: {max([ann['area'] for ann in anns])} pixels")

def analyze_dataset(img_dir, ann_dir, train_file, val_file):
    """データセット全体の統計情報を分析"""
    print("\n=== データセット分析 ===\n")
    
    # ファイルリストを読み込み
    with open(train_file, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    with open(val_file, 'r') as f:
        val_names = [line.strip() for line in f if line.strip()]
    
    print(f"学習データ: {len(train_names)}枚")
    print(f"検証データ: {len(val_names)}枚")
    print(f"合計: {len(train_names) + len(val_names)}枚")
    
    # サンプリングして詳細分析（全データは時間がかかるため）
    sample_size = min(100, len(train_names))
    sampled_names = random.sample(train_names, sample_size)
    
    total_instances = 0
    all_areas = []
    all_num_instances = []
    errors_count = 0
    
    print(f"\n{sample_size}枚のサンプルを分析中...")
    for name in tqdm(sampled_names):
        stem = name.rsplit('.', 1)[0]
        ann_path = Path(ann_dir) / f"{stem}.json"
        
        if not ann_path.exists():
            errors_count += 1
            continue
        
        errors, data = verify_json_structure(ann_path)
        if errors:
            errors_count += 1
            continue
        
        anns = data["annotations"]
        total_instances += len(anns)
        all_num_instances.append(len(anns))
        all_areas.extend([ann["area"] for ann in anns])
    
    if all_num_instances:
        print(f"\n=== 統計情報（{sample_size}枚のサンプルより） ===")
        print(f"エラーファイル数: {errors_count}")
        print(f"平均インスタンス数/画像: {np.mean(all_num_instances):.1f}")
        print(f"最小インスタンス数/画像: {min(all_num_instances)}")
        print(f"最大インスタンス数/画像: {max(all_num_instances)}")
        print(f"平均インスタンス面積: {np.mean(all_areas):.0f} pixels")
        print(f"中央値インスタンス面積: {np.median(all_areas):.0f} pixels")

def main():
    base_dir = "data/foodmix_sa1b"
    img_dir = os.path.join(base_dir, "images")
    ann_dir = os.path.join(base_dir, "annotations")
    train_file = os.path.join(base_dir, "train.txt")
    val_file = os.path.join(base_dir, "val.txt")
    
    # パスの存在確認
    if not all(os.path.exists(p) for p in [img_dir, ann_dir, train_file, val_file]):
        print("エラー: 必要なディレクトリまたはファイルが見つかりません")
        print(f"  画像ディレクトリ: {os.path.exists(img_dir)}")
        print(f"  アノテーションディレクトリ: {os.path.exists(ann_dir)}")
        print(f"  学習リスト: {os.path.exists(train_file)}")
        print(f"  検証リスト: {os.path.exists(val_file)}")
        return
    
    # データセット分析
    analyze_dataset(img_dir, ann_dir, train_file, val_file)
    
    # サンプル可視化（3枚）
    print("\n=== サンプル可視化 ===")
    for i in range(3):
        print(f"\nサンプル {i+1}/3:")
        visualize_sample(img_dir, ann_dir)
        print("-" * 50)

if __name__ == "__main__":
    main()