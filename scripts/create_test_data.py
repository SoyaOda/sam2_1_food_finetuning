#!/usr/bin/env python3
"""
テスト用の最小限のデータセットを作成
SAM2の学習が動作するか確認するため
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from pycocotools import mask as mask_utils


def create_test_image(size=(512, 512), num_objects=3):
    """テスト用の画像とマスクを生成"""
    # ランダムな背景色
    img = Image.new('RGB', size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # いくつかの矩形を描画
    masks = []
    for i in range(num_objects):
        x1 = np.random.randint(0, size[0] - 100)
        y1 = np.random.randint(0, size[1] - 100)
        x2 = x1 + np.random.randint(50, 100)
        y2 = y1 + np.random.randint(50, 100)
        
        # 画像に色付き矩形を描画
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # マスクを作成
        mask = np.zeros(size[::-1], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        masks.append({
            'mask': mask,
            'bbox': [x1, y1, x2-x1, y2-y1]
        })
    
    return img, masks


def rle_encode(mask):
    """マスクをRLE形式にエンコード"""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def create_test_dataset(output_dir, num_images=5):
    """テスト用データセットを作成"""
    output_path = Path(output_dir)
    img_dir = output_path / "images"
    ann_dir = output_path / "annotations"
    
    # ディレクトリ作成
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    file_list = []
    
    for i in range(num_images):
        # 画像とマスクを生成
        img, masks = create_test_image()
        
        # ファイル名
        img_name = f"test_{i:04d}.jpg"
        file_list.append(img_name)
        
        # 画像を保存
        img_path = img_dir / img_name
        img.save(img_path)
        
        # アノテーションを作成
        h, w = img.size[::-1]
        annotations = []
        for j, mask_data in enumerate(masks):
            mask = mask_data['mask']
            bbox = mask_data['bbox']
            
            ann = {
                "id": j,
                "segmentation": rle_encode(mask),
                "bbox": bbox,
                "area": int(mask.sum()),
                "predicted_iou": 1.0,
                "stability_score": 1.0,
                "crop_box": [0, 0, w, h],
                "point_coords": [[int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)]]
            }
            annotations.append(ann)
        
        # JSONとして保存
        ann_data = {
            "image": {
                "image_id": i,
                "width": w,
                "height": h,
                "file_name": img_name
            },
            "annotations": annotations
        }
        
        ann_path = ann_dir / f"test_{i:04d}.json"
        with open(ann_path, 'w') as f:
            json.dump(ann_data, f)
    
    # train.txtとval.txtを作成
    split_idx = int(num_images * 0.8)
    train_files = file_list[:split_idx]
    val_files = file_list[split_idx:]
    
    with open(output_path / "train.txt", 'w') as f:
        for fname in train_files:
            f.write(f"{fname}\n")
    
    with open(output_path / "val.txt", 'w') as f:
        for fname in val_files:
            f.write(f"{fname}\n")
    
    print(f"テストデータセット作成完了:")
    print(f"  場所: {output_path}")
    print(f"  画像数: {num_images}")
    print(f"  訓練: {len(train_files)}")
    print(f"  検証: {len(val_files)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="テスト用データセット作成")
    parser.add_argument("--output", type=str, default="data/test_dataset", 
                        help="出力ディレクトリ")
    parser.add_argument("--num-images", type=int, default=10,
                        help="生成する画像数")
    
    args = parser.parse_args()
    
    create_test_dataset(args.output, args.num_images)