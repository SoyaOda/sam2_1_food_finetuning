#!/usr/bin/env python
"""
FoodSeg103データセットをSA-1B形式に変換
セマンティックマスクから連結成分ごとのインスタンスマスクに分解
"""

import os
import json
import glob
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from pycocotools import mask as mask_utils
from tqdm import tqdm
from pathlib import Path

def rle_of_binary_mask(bin_mask: np.ndarray):
    """バイナリマスクをRLE形式にエンコード（Fortran order）"""
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def extract_instances_from_semantic(mask_arr: np.ndarray, ignore_ids={0}):
    """セマンティックマスクから連結成分ごとにインスタンスを抽出"""
    h, w = mask_arr.shape
    ann_list = []
    ann_id = 0
    valid_ids = [i for i in np.unique(mask_arr) if i not in ignore_ids]
    
    for cid in valid_ids:
        # 各クラスIDのマスクを取得
        comp = (mask_arr == cid).astype(np.uint8)
        # 連結成分ラベリング
        lab = label(comp, connectivity=1, background=0)
        
        for lid in np.unique(lab):
            if lid == 0:  # 背景スキップ
                continue
            
            # インスタンスマスク
            inst = (lab == lid)
            area = int(inst.sum())
            
            # 極小領域を除外
            if area < 10:
                continue
            
            # バウンディングボックス計算
            ys, xs = np.where(inst)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            bbox = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
            
            # RLEエンコード
            rle = rle_of_binary_mask(inst)
            
            # SA-1B形式のアノテーション
            ann_list.append({
                "id": ann_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
                # SA-1Bと互換の補助項目
                "predicted_iou": 1.0,
                "stability_score": 1.0,
                "crop_box": [0, 0, w, h],
                "point_coords": [[int((x0+x1)/2), int((y0+y1)/2)]],  # 中心点
            })
            ann_id += 1
    
    return ann_list

def main():
    # FoodSeg103データのパス
    root = Path("data/FoodSeg103")
    
    # 画像とマスクのディレクトリ
    train_img_dir = root / "Images" / "train"
    train_mask_dir = root / "Masks" / "train"
    val_img_dir = root / "Images" / "validation"
    val_mask_dir = root / "Masks" / "validation"
    
    # 出力ディレクトリ
    out_root = Path("data/foodmix_sa1b")
    out_img_dir = out_root / "images"
    out_ann_dir = out_root / "annotations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルリスト用
    train_files = []
    val_files = []
    
    # 学習データの処理
    print("学習データを処理中...")
    if train_img_dir.exists():
        img_paths = sorted(train_img_dir.glob("*.jpg"))
        
        for img_path in tqdm(img_paths, desc="Train"):
            img_name = img_path.name
            stem = img_path.stem
            
            # 対応するマスクファイルを探す
            mask_path = train_mask_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            
            # 画像とマスクを読み込み
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
            mask_arr = np.array(mask)
            
            # RGBマスクの場合、最初のチャンネルを使用
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            
            h, w = mask_arr.shape
            
            # インスタンスを抽出
            anns = extract_instances_from_semantic(mask_arr, ignore_ids={0})
            
            if len(anns) == 0:
                continue
            
            # SA-1B形式のJSON
            out = {
                "image": {
                    "image_id": 0,
                    "width": int(w),
                    "height": int(h),
                    "file_name": img_name
                },
                "annotations": anns
            }
            
            # ファイル名を変更（train_プレフィックスを追加）
            new_img_name = f"train_{stem}.jpg"
            new_ann_name = f"train_{stem}.json"
            
            # 画像をコピー保存
            img.save(out_img_dir / new_img_name)
            
            # アノテーションを保存
            with open(out_ann_dir / new_ann_name, "w") as f:
                json.dump(out, f)
            
            train_files.append(new_img_name)
    
    # 検証データの処理
    print("\n検証データを処理中...")
    if val_img_dir.exists():
        img_paths = sorted(val_img_dir.glob("*.jpg"))
        
        for img_path in tqdm(img_paths, desc="Validation"):
            img_name = img_path.name
            stem = img_path.stem
            
            # 対応するマスクファイルを探す
            mask_path = val_mask_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            
            # 画像とマスクを読み込み
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
            mask_arr = np.array(mask)
            
            # RGBマスクの場合、最初のチャンネルを使用
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            
            h, w = mask_arr.shape
            
            # インスタンスを抽出
            anns = extract_instances_from_semantic(mask_arr, ignore_ids={0})
            
            if len(anns) == 0:
                continue
            
            # SA-1B形式のJSON
            out = {
                "image": {
                    "image_id": 0,
                    "width": int(w),
                    "height": int(h),
                    "file_name": img_name
                },
                "annotations": anns
            }
            
            # ファイル名を変更（val_プレフィックスを追加）
            new_img_name = f"val_{stem}.jpg"
            new_ann_name = f"val_{stem}.json"
            
            # 画像をコピー保存
            img.save(out_img_dir / new_img_name)
            
            # アノテーションを保存
            with open(out_ann_dir / new_ann_name, "w") as f:
                json.dump(out, f)
            
            val_files.append(new_img_name)
    
    # train.txtとval.txtを作成
    print("\nファイルリストを作成中...")
    with open(out_root / "train.txt", "w") as f:
        f.write("\n".join(train_files))
    
    with open(out_root / "val.txt", "w") as f:
        f.write("\n".join(val_files))
    
    print(f"\n処理完了:")
    print(f"  学習データ: {len(train_files)}枚")
    print(f"  検証データ: {len(val_files)}枚")
    print(f"  保存先: {out_root}")

if __name__ == "__main__":
    main()