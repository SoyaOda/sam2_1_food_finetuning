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
    """
    バイナリマスクをRLE形式にエンコード（Fortran order）
    pycocotoolsの環境依存問題に対応
    """
    if bin_mask.dtype != np.uint8:
        bin_mask = bin_mask.astype(np.uint8)
    
    # Fortran order (列優先) でエンコード - SA-1B形式の要件
    encoded = mask_utils.encode(np.asfortranarray(bin_mask[:, :, None]))
    
    # pycocotoolsの結果が辞書かリストかをチェック
    if isinstance(encoded, dict):
        # 通常の場合（辞書）
        if isinstance(encoded.get("counts"), bytes):
            encoded["counts"] = encoded["counts"].decode("ascii")
        return encoded
    elif isinstance(encoded, list) and len(encoded) == 1:
        # リストの場合（一部のpycocotoolsバージョン）
        rle_dict = encoded[0]
        if isinstance(rle_dict, dict) and "counts" in rle_dict:
            if isinstance(rle_dict["counts"], bytes):
                rle_dict["counts"] = rle_dict["counts"].decode("ascii")
            return rle_dict
        else:
            # 手動でRLE辞書を作成
            return {
                "size": [bin_mask.shape[0], bin_mask.shape[1]],
                "counts": str(encoded)  # フォールバック
            }
    else:
        # その他のケース、手動でRLE辞書を作成
        return {
            "size": [bin_mask.shape[0], bin_mask.shape[1]], 
            "counts": str(encoded)  # フォールバック
        }

def extract_instances_from_semantic(mask_arr: np.ndarray, ignore_ids={0}, min_area=32):
    """
    セマンティックマスクから連結成分ごとにインスタンスを抽出
    o3-query回答の改善を適用（8連結、ノイズ除去、エラーハンドリング強化）
    """
    from skimage.morphology import remove_small_objects
    
    h, w = mask_arr.shape
    ann_list = []
    ann_id = 0
    valid_ids = [i for i in np.unique(mask_arr) if i not in ignore_ids]
    
    for cid in valid_ids:
        try:
            # 各クラスIDのバイナリマスクを取得
            bin_mask = (mask_arr == cid).astype(bool)
            
            # 微小ノイズ除去（連結成分分析の前に実行）
            cleaned = remove_small_objects(bin_mask, min_size=min_area)
            
            # 連結成分ラベリング（8連結に変更）
            lab = label(cleaned.astype(np.uint8), connectivity=2, background=0)
            
            for lid in np.unique(lab):
                if lid == 0:  # 背景スキップ
                    continue
                
                # インスタンスマスク
                inst = (lab == lid)
                area = int(inst.sum())
                
                # 最小領域チェック
                if area < min_area:
                    continue
                
                # バウンディングボックス計算
                ys, xs = np.where(inst)
                if len(ys) == 0 or len(xs) == 0:
                    continue
                    
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
                
                # 中心点計算
                center_x = int(x0 + (x1 - x0) / 2)
                center_y = int(y0 + (y1 - y0) / 2)
                
                # RLEエンコード
                rle = rle_of_binary_mask(inst)
                
                # SA-1B形式のアノテーション（修正版）
                ann_list.append({
                    "id": ann_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": area,
                    # SA-1B互換の補助フィールド
                    "predicted_iou": 1.0,
                    "stability_score": 1.0,
                    "crop_box": [0.0, 0.0, float(w), float(h)],
                    "point_coords": [[center_x, center_y]],
                    # 任意のメタ（学習ローダは無視/通過）
                    "category_id": int(cid)
                })
                ann_id += 1
                
        except Exception as e:
            print(f"クラス{cid}の処理中にエラー: {e}")
            continue
    
    return ann_list

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FoodSeg103をSA-1B形式に変換（修正版）")
    parser.add_argument("--input", type=str, default="data/FoodSeg103", help="FoodSeg103データセットのルートディレクトリ")
    parser.add_argument("--output", type=str, default="data/foodmix_sa1b", help="出力ディレクトリ") 
    parser.add_argument("--min-area", type=int, default=32, help="最小インスタンスサイズ（ピクセル）")
    args = parser.parse_args()
    
    # FoodSeg103データのパス
    root = Path(args.input)
    min_area = args.min_area
    
    # 画像とマスクのディレクトリ
    train_img_dir = root / "Images" / "train"
    train_mask_dir = root / "Masks" / "train"
    val_img_dir = root / "Images" / "validation"
    val_mask_dir = root / "Masks" / "validation"
    
    # 出力ディレクトリ
    out_root = Path(args.output)
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
            
            # ファイル名を事前に定義（train_プレフィックスを追加）
            new_img_name = f"train_{stem}.jpg"
            new_ann_name = f"train_{stem}.json"
            
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
            
            try:
                # インスタンスを抽出（min_area=32でノイズ除去強化）
                anns = extract_instances_from_semantic(mask_arr, ignore_ids={0}, min_area=min_area)
                
                if len(anns) == 0:
                    print(f"警告: {stem} - 有効なインスタンスが見つかりません")
                    continue
                
                # ユニークなimage_idを生成
                image_id = int(stem) if stem.isdigit() else hash(stem) % 2**31
                
                # SA-1B形式のJSON（修正版）
                out = {
                    "image": {
                        "image_id": image_id,
                        "width": int(w),
                        "height": int(h),
                        "file_name": new_img_name  # 実際の出力ファイル名
                    },
                    "annotations": anns
                }
            except Exception as e:
                print(f"エラー: {stem} の処理中にエラーが発生: {e}")
                continue
            

            
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
            
            # ファイル名を事前に定義（val_プレフィックスを追加）
            new_img_name = f"val_{stem}.jpg"
            new_ann_name = f"val_{stem}.json"
            
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
            
            try:
                # インスタンスを抽出（min_area=32でノイズ除去強化）
                anns = extract_instances_from_semantic(mask_arr, ignore_ids={0}, min_area=min_area)
                
                if len(anns) == 0:
                    print(f"警告: {stem} - 有効なインスタンスが見つかりません")
                    continue
                
                # ユニークなimage_idを生成
                image_id = int(stem) if stem.isdigit() else hash(stem) % 2**31
                
                # SA-1B形式のJSON（修正版）
                out = {
                    "image": {
                        "image_id": image_id,
                        "width": int(w),
                        "height": int(h),
                        "file_name": new_img_name  # 実際の出力ファイル名
                    },
                    "annotations": anns
                }
            except Exception as e:
                print(f"エラー: {stem} の処理中にエラーが発生: {e}")
                continue
            

            
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