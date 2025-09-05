#!/usr/bin/env python3
"""
13_optional_resize_1024.py - 画像を1024x1024にリサイズ（オプション）

SAM2.1の学習を安定させるため、全画像を1024x1024の固定サイズにリサイズする。
アスペクト比は保持せず、ストレッチして変形させる。
"""

import os
import glob
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import json
import shutil


def resize_images(
    data_dir: str,
    target_size: int = 1024,
    backup: bool = True,
    quality: int = 95
):
    """
    画像を指定サイズにリサイズ
    
    Args:
        data_dir: foodmix_sa1bディレクトリのパス
        target_size: リサイズ後のサイズ（正方形）
        backup: オリジナル画像のバックアップを作成するか
        quality: JPEG保存時の品質（1-100）
    """
    data_path = Path(data_dir)
    img_dir = data_path / "images"
    
    if not img_dir.exists():
        print(f"エラー: {img_dir} が見つかりません")
        print("前処理スクリプトを先に実行してください")
        return
    
    # バックアップディレクトリ作成
    if backup:
        backup_dir = data_path / "images_original"
        if not backup_dir.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"バックアップディレクトリを作成: {backup_dir}")
    
    # 画像ファイル一覧取得
    img_files = sorted(
        list(img_dir.glob("*.jpg")) + 
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )
    
    if not img_files:
        print("処理する画像が見つかりません")
        return
    
    print(f"=== 画像リサイズ処理 ===")
    print(f"対象画像数: {len(img_files)}")
    print(f"目標サイズ: {target_size}x{target_size}")
    print(f"バックアップ: {'有効' if backup else '無効'}")
    
    # 統計情報収集用
    original_sizes = []
    skipped_count = 0
    resized_count = 0
    error_count = 0
    
    for img_path in tqdm(img_files, desc="画像リサイズ中"):
        try:
            # 画像読み込み
            img = Image.open(img_path)
            original_size = img.size
            original_sizes.append(original_size)
            
            # すでに目標サイズの場合はスキップ
            if original_size == (target_size, target_size):
                skipped_count += 1
                continue
            
            # バックアップ作成
            if backup:
                backup_path = backup_dir / img_path.name
                if not backup_path.exists():
                    shutil.copy2(img_path, backup_path)
            
            # RGBに変換（PNGのアルファチャンネル対応）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # リサイズ（アスペクト比無視でストレッチ）
            resized_img = img.resize(
                (target_size, target_size),
                Image.Resampling.LANCZOS  # 高品質リサンプリング
            )
            
            # 保存（元のファイルを上書き）
            if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                resized_img.save(img_path, 'JPEG', quality=quality, optimize=True)
            else:
                # PNGの場合はJPEGに変換して保存
                new_path = img_path.with_suffix('.jpg')
                resized_img.save(new_path, 'JPEG', quality=quality, optimize=True)
                
                # 元のPNGファイルを削除
                if new_path != img_path:
                    img_path.unlink()
                    
                    # 対応するアノテーションファイルの更新が必要
                    update_annotation_filename(data_path, img_path.name, new_path.name)
            
            resized_count += 1
            
        except Exception as e:
            print(f"\nエラー: {img_path.name} の処理中にエラーが発生: {e}")
            error_count += 1
            continue
    
    # 統計情報表示
    print(f"\n=== リサイズ処理完了 ===")
    print(f"処理済み: {resized_count} ファイル")
    print(f"スキップ: {skipped_count} ファイル（既に{target_size}x{target_size}）")
    print(f"エラー: {error_count} ファイル")
    
    if original_sizes:
        # オリジナルサイズの統計
        widths = [s[0] for s in original_sizes]
        heights = [s[1] for s in original_sizes]
        
        print(f"\n=== オリジナル画像サイズ統計 ===")
        print(f"幅: 最小={min(widths)}, 最大={max(widths)}, 平均={sum(widths)/len(widths):.1f}")
        print(f"高さ: 最小={min(heights)}, 最大={max(heights)}, 平均={sum(heights)/len(heights):.1f}")
        
        # アスペクト比の分布
        aspect_ratios = [w/h for w, h in original_sizes]
        print(f"アスペクト比: 最小={min(aspect_ratios):.2f}, 最大={max(aspect_ratios):.2f}, 平均={sum(aspect_ratios)/len(aspect_ratios):.2f}")
    
    if backup and resized_count > 0:
        print(f"\nオリジナル画像のバックアップ: {backup_dir}")
        print("復元が必要な場合は以下を実行:")
        print(f"  cp -r {backup_dir}/* {img_dir}/")


def update_annotation_filename(data_dir: Path, old_filename: str, new_filename: str):
    """
    アノテーションJSON内のファイル名を更新（PNG→JPG変換時）
    """
    ann_dir = data_dir / "annotations"
    stem = Path(old_filename).stem
    ann_path = ann_dir / f"{stem}.json"
    
    if ann_path.exists():
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
            
            if "image" in data and "file_name" in data["image"]:
                data["image"]["file_name"] = new_filename
                
                with open(ann_path, "w") as f:
                    json.dump(data, f)
        except Exception as e:
            print(f"警告: {ann_path} の更新に失敗: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="画像を1024x1024にリサイズ（SAM2.1学習の安定化のため）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/foodmix_sa1b",
        help="SA-1B形式のデータが格納されているディレクトリ"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="リサイズ後のサイズ（デフォルト: 1024）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="オリジナル画像のバックアップを作成しない"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG保存品質（1-100、デフォルト: 95）"
    )
    
    args = parser.parse_args()
    
    # 確認プロンプト
    if not args.no_backup:
        print("注意: 画像をリサイズすると元のアスペクト比が失われます")
        print("オリジナル画像はバックアップされます")
    else:
        print("警告: バックアップなしで画像を上書きします")
        response = input("続行しますか？ (y/n): ")
        if response.lower() != 'y':
            print("処理を中止しました")
            return
    
    # 処理実行
    resize_images(
        args.data_dir,
        args.size,
        backup=not args.no_backup,
        quality=args.quality
    )
    
    print("\n=== 次のステップ ===")
    print("1. 学習設定YAMLを確認: configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml")
    print("2. 学習を開始: bash scripts/20_train_food_sam2.sh")


if __name__ == "__main__":
    main()