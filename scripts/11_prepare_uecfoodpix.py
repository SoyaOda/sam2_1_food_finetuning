#!/usr/bin/env python3
"""
11_prepare_uecfoodpix.py - UEC-FoodPix CompleteデータセットをSA-1B形式に変換

UEC-FoodPix Completeのセマンティックマスク（Rチャネルにクラス情報）を
連結成分ごとのインスタンスマスクに分解し、SA-1B互換のJSON形式で保存する。
"""

import os
import json
import glob
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from skimage.measure import label
from tqdm import tqdm
import argparse
from pathlib import Path


def rle_of_binary_mask(bin_mask: np.ndarray):
    """
    バイナリマスクをRLE（Run-Length Encoding）形式に変換
    SA-1B形式のため、Fortran orderでエンコード
    """
    # Fortran order (列優先) でエンコード - SA-1B形式の要件
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    # countをASCII文字列に変換（JSON互換性のため）
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def extract_instances_from_uec_mask(mask_rgb: np.ndarray):
    """
    UEC-FoodPixのマスク（RチャネルにクラスID）からインスタンスを抽出
    
    Args:
        mask_rgb: RGB形式のマスク画像（HxWx3）
    
    Returns:
        SA-1B形式のアノテーションリスト
    """
    # UEC-FoodPixの仕様: RチャネルにクラスIDが格納されている
    if mask_rgb.ndim == 3:
        mask_arr = mask_rgb[:, :, 0]  # Rチャネルのみ使用
    else:
        mask_arr = mask_rgb
    
    H, W = mask_arr.shape
    anns = []
    ann_id = 0
    
    # ユニークなクラスIDを取得（背景0を除く）
    unique_ids = np.unique(mask_arr)
    valid_ids = [cid for cid in unique_ids if cid != 0]
    
    for cid in valid_ids:
        # 各クラスのピクセルを抽出
        class_mask = (mask_arr == cid).astype(np.uint8)
        
        # 連結成分ラベリング（8連結）
        labeled_mask = label(class_mask, connectivity=2, background=0)
        
        # 各連結成分をインスタンスとして処理
        for lid in np.unique(labeled_mask):
            if lid == 0:  # 背景スキップ
                continue
            
            # インスタンスマスク生成
            inst_mask = (labeled_mask == lid)
            area = int(inst_mask.sum())
            
            # 極小領域の除外（ノイズ除去）
            if area < 10:
                continue
            
            # バウンディングボックス計算
            ys, xs = np.where(inst_mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]  # [x, y, width, height]
            
            # RLEエンコード
            rle = rle_of_binary_mask(inst_mask)
            
            # SA-1B形式のアノテーション
            anns.append({
                "id": ann_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
                # SA-1B互換の補助フィールド
                "predicted_iou": 1.0,  # GT なので 1.0
                "stability_score": 1.0,  # GT なので 1.0
                "crop_box": [0, 0, W, H],  # フル画像
                "point_coords": [[int((x0+x1)/2), int((y0+y1)/2)]],  # 中心点
            })
            ann_id += 1
    
    return anns


def find_uec_data_structure(root_dir):
    """
    UEC-FoodPix Completeのディレクトリ構造を自動検出
    """
    root_path = Path(root_dir)
    
    # 可能なディレクトリ構造パターン
    base_patterns = [
        "data/UECFoodPIXCOMPLETE",
        "data/UECFOODPIXCOMPLETE",
        "UECFoodPIXCOMPLETE",
        "UECFOODPIXCOMPLETE",
    ]
    
    for pattern in base_patterns:
        base_dir = root_path / pattern
        if base_dir.exists():
            train_img = base_dir / "train" / "img"
            train_mask = base_dir / "train" / "mask"
            test_img = base_dir / "test" / "img"
            test_mask = base_dir / "test" / "mask"
            
            if train_img.exists() and train_mask.exists():
                return {
                    "train_img": train_img,
                    "train_mask": train_mask,
                    "test_img": test_img if test_img.exists() else None,
                    "test_mask": test_mask if test_mask.exists() else None,
                    "base_dir": base_dir
                }
    
    # 見つからない場合は glob で探索
    train_dirs = list(root_path.glob("**/train/img"))
    if train_dirs:
        train_img = train_dirs[0]
        train_mask = train_img.parent / "mask"
        base_dir = train_img.parent.parent
        test_img = base_dir / "test" / "img"
        test_mask = base_dir / "test" / "mask"
        
        return {
            "train_img": train_img,
            "train_mask": train_mask if train_mask.exists() else None,
            "test_img": test_img if test_img.exists() else None,
            "test_mask": test_mask if test_mask.exists() else None,
            "base_dir": base_dir
        }
    
    return None


def process_uec_split(img_dir, mask_dir, output_dir, split_name="train"):
    """
    UEC-FoodPixの1つのスプリット（trainまたはtest）を処理
    """
    processed_pairs = []
    
    # 出力ディレクトリ
    out_img_dir = Path(output_dir) / "images"
    out_ann_dir = Path(output_dir) / "annotations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル一覧取得
    img_paths = sorted(img_dir.glob("*.jpg"))
    
    print(f"\n{split_name}データ処理中...")
    print(f"画像数: {len(img_paths)}")
    
    for img_path in tqdm(img_paths, desc=f"UEC-{split_name}処理中"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        
        if not mask_path.exists():
            print(f"警告: {img_path.name} に対応するマスクが見つかりません")
            continue
        
        try:
            # 画像とマスクを読み込み
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
            mask_arr = np.array(mask)
            
            # UEC-FoodPixはRチャネルにクラスID
            if mask_arr.ndim != 3:
                print(f"警告: {mask_path.name} が期待される形式（RGB）ではありません")
                continue
            
            # インスタンス抽出
            anns = extract_instances_from_uec_mask(mask_arr)
            
            if len(anns) == 0:
                # アノテーションがない画像もスキップしない（背景のみの画像として保持）
                print(f"情報: {img_path.name} にインスタンスが見つかりません（背景のみ）")
            
            # SA-1B形式のJSON作成
            H, W = mask_arr.shape[:2]
            sa1b_data = {
                "image": {
                    "image_id": 0,
                    "width": int(W),
                    "height": int(H),
                    "file_name": f"uec_{split_name}_{img_path.name}"  # プレフィックス追加
                },
                "annotations": anns
            }
            
            # ファイル保存（ファイル名にプレフィックスを追加して衝突を避ける）
            out_img_name = f"uec_{split_name}_{img_path.name}"
            out_img_path = out_img_dir / out_img_name
            out_json_path = out_ann_dir / f"uec_{split_name}_{stem}.json"
            
            img.save(out_img_path)
            with open(out_json_path, "w") as f:
                json.dump(sa1b_data, f)
            
            processed_pairs.append(out_img_name)
            
        except Exception as e:
            print(f"エラー: {img_path.name} の処理中にエラーが発生: {e}")
            continue
    
    return processed_pairs


def process_uecfoodpix(root_dir, output_dir):
    """
    UEC-FoodPix Completeデータセット全体を処理
    """
    print("UEC-FoodPix Completeデータセットの処理を開始...")
    
    # ディレクトリ構造検出
    data_structure = find_uec_data_structure(root_dir)
    
    if data_structure is None:
        print(f"エラー: {root_dir} にUEC-FoodPixデータが見つかりません")
        print("ディレクトリ構造を確認してください")
        print("期待される構造: data/UECFoodPIXCOMPLETE/train/img, data/UECFoodPIXCOMPLETE/train/mask")
        return
    
    print(f"ベースディレクトリ: {data_structure['base_dir']}")
    
    all_processed = []
    
    # trainデータ処理
    if data_structure["train_img"] and data_structure["train_mask"]:
        train_processed = process_uec_split(
            data_structure["train_img"],
            data_structure["train_mask"],
            output_dir,
            "train"
        )
        all_processed.extend(train_processed)
        print(f"trainデータ処理完了: {len(train_processed)} ペア")
    
    # testデータ処理（存在する場合）
    if data_structure["test_img"] and data_structure["test_mask"]:
        test_processed = process_uec_split(
            data_structure["test_img"],
            data_structure["test_mask"],
            output_dir,
            "test"
        )
        all_processed.extend(test_processed)
        print(f"testデータ処理完了: {len(test_processed)} ペア")
    
    # カテゴリ情報のコピー（参考用）
    category_file = data_structure["base_dir"] / "category.txt"
    if category_file.exists():
        output_category = Path(output_dir) / "uec_category.txt"
        with open(category_file, "r") as f_in:
            with open(output_category, "w") as f_out:
                f_out.write("# UEC-FoodPix Complete カテゴリ情報\n")
                f_out.write(f_in.read())
        print(f"カテゴリ情報をコピー: {output_category}")
    
    print(f"\n処理完了: 合計 {len(all_processed)} ペア")
    print(f"出力先: {output_dir}")
    
    return all_processed


def main():
    parser = argparse.ArgumentParser(description="UEC-FoodPix CompleteをSA-1B形式に変換")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/UECFOODPIXCOMPLETE",
        help="UEC-FoodPix Completeデータセットのルートディレクトリ"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/foodmix_sa1b",
        help="出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    # 処理実行
    process_uecfoodpix(args.input, args.output)
    
    print("\nUEC-FoodPix Completeの前処理が完了しました！")
    print("次のステップ:")
    print("1. python scripts/12_merge_to_sa1b.py - データセットの統合とスプリット作成")
    print("2. python scripts/13_optional_resize_1024.py - （任意）画像の1024x1024リサイズ")


if __name__ == "__main__":
    main()