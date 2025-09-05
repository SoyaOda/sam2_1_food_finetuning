#!/usr/bin/env python3
"""
21_eval_and_viz.py - 学習済みSAM2.1モデルの評価と可視化

微調整済みのSAM2.1モデルを使用して、
検証データに対する推論と可視化を行う。
"""

import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm

# SAM2インポート
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("エラー: SAM2がインポートできません")
    print("external/sam2ディレクトリでpip install -e .を実行してください")
    exit(1)

# pycocotoolsインポート
try:
    from pycocotools import mask as mask_utils
except ImportError:
    print("エラー: pycocotoolsがインポートできません")
    print("pip install pycocotools を実行してください")
    exit(1)


def rle_to_binary_mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """
    RLEエンコードされたマスクをバイナリマスクに変換
    """
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    
    mask = mask_utils.decode(rle)
    return mask.reshape((height, width), order="F")


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    2つのマスク間のIoU（Intersection over Union）を計算
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_single_image(
    predictor: SAM2ImagePredictor,
    img_path: Path,
    ann_path: Path,
    use_points: bool = True,
    visualize: bool = False
) -> Dict:
    """
    単一画像に対する評価
    """
    # 画像読み込み
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    
    # アノテーション読み込み
    with open(ann_path, "r") as f:
        ann_data = json.load(f)
    
    gt_annotations = ann_data.get("annotations", [])
    
    if len(gt_annotations) == 0:
        return {"iou": 0.0, "num_gt": 0, "num_pred": 0}
    
    # 画像をpredictorにセット
    predictor.set_image(img)
    
    # 予測マスク生成
    pred_masks = []
    
    if use_points:
        # GTのポイントを使用して予測
        for gt_ann in gt_annotations:
            if "point_coords" in gt_ann and gt_ann["point_coords"]:
                point_coords = np.array(gt_ann["point_coords"])
                point_labels = np.ones(len(point_coords))
                
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )
                
                if len(masks) > 0:
                    pred_masks.append(masks[0])
    else:
        # 自動マスク生成（ポイントなし）
        # この場合はSAM2AutomaticMaskGeneratorを使用
        pass  # 簡略化のため省略
    
    # GTマスクをバイナリ形式に変換
    gt_masks = []
    for gt_ann in gt_annotations:
        if "segmentation" in gt_ann:
            gt_mask = rle_to_binary_mask(gt_ann["segmentation"], H, W)
            gt_masks.append(gt_mask)
    
    # IoU計算（最良マッチング）
    ious = []
    for gt_mask in gt_masks:
        best_iou = 0.0
        for pred_mask in pred_masks:
            iou = compute_iou(gt_mask, pred_mask)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    # 可視化
    if visualize:
        visualize_results(img, gt_masks, pred_masks, mean_iou)
    
    return {
        "iou": mean_iou,
        "num_gt": len(gt_masks),
        "num_pred": len(pred_masks),
        "ious": ious
    }


def visualize_results(
    img: np.ndarray,
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    iou: float
):
    """
    結果の可視化
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元画像
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # GT マスク
    axes[1].imshow(img)
    for mask in gt_masks:
        show_mask(mask, axes[1], random_color=True)
    axes[1].set_title(f"Ground Truth ({len(gt_masks)} masks)")
    axes[1].axis("off")
    
    # 予測マスク
    axes[2].imshow(img)
    for mask in pred_masks:
        show_mask(mask, axes[2], random_color=True)
    axes[2].set_title(f"Predictions (IoU: {iou:.3f})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()


def show_mask(mask, ax, random_color=False):
    """
    マスクを可視化
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def evaluate_dataset(
    checkpoint_path: str,
    config_path: str,
    data_dir: str,
    split: str = "val",
    num_samples: int = None,
    visualize: bool = False,
    device: str = "cuda"
):
    """
    データセット全体の評価
    """
    print(f"=== SAM2.1 評価開始 ===")
    print(f"チェックポイント: {checkpoint_path}")
    print(f"設定: {config_path}")
    print(f"データ: {data_dir}")
    print(f"スプリット: {split}")
    
    # デバイス設定
    if not torch.cuda.is_available() and device == "cuda":
        print("警告: CUDAが利用できません。CPUで実行します")
        device = "cpu"
    
    # モデル構築
    print("\nモデルをロード中...")
    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("モデルロード完了")
    
    # データパス設定
    data_path = Path(data_dir)
    img_dir = data_path / "images"
    ann_dir = data_path / "annotations"
    split_file = data_path / f"{split}.txt"
    
    # ファイルリスト読み込み
    if not split_file.exists():
        print(f"エラー: {split_file} が見つかりません")
        return
    
    with open(split_file, "r") as f:
        file_list = [line.strip() for line in f if line.strip()]
    
    if num_samples:
        file_list = random.sample(file_list, min(num_samples, len(file_list)))
    
    print(f"\n評価対象: {len(file_list)} ファイル")
    
    # 評価実行
    results = []
    for fname in tqdm(file_list, desc="評価中"):
        stem = Path(fname).stem
        img_path = img_dir / fname
        ann_path = ann_dir / f"{stem}.json"
        
        if not img_path.exists() or not ann_path.exists():
            print(f"警告: {fname} のデータが不完全です")
            continue
        
        try:
            result = evaluate_single_image(
                predictor,
                img_path,
                ann_path,
                use_points=True,
                visualize=visualize and len(results) < 5  # 最初の5件のみ可視化
            )
            results.append(result)
        except Exception as e:
            print(f"エラー: {fname} の処理中にエラー: {e}")
            continue
    
    # 統計計算
    if results:
        mean_iou = np.mean([r["iou"] for r in results])
        total_gt = sum([r["num_gt"] for r in results])
        total_pred = sum([r["num_pred"] for r in results])
        
        # IoU分布
        all_ious = []
        for r in results:
            if "ious" in r:
                all_ious.extend(r["ious"])
        
        print(f"\n=== 評価結果 ===")
        print(f"平均IoU: {mean_iou:.3f}")
        print(f"総GTマスク数: {total_gt}")
        print(f"総予測マスク数: {total_pred}")
        
        if all_ious:
            print(f"\nIoU分布:")
            print(f"  最小: {min(all_ious):.3f}")
            print(f"  25%: {np.percentile(all_ious, 25):.3f}")
            print(f"  50%: {np.percentile(all_ious, 50):.3f}")
            print(f"  75%: {np.percentile(all_ious, 75):.3f}")
            print(f"  最大: {max(all_ious):.3f}")
            
            # IoUヒストグラム
            plt.figure(figsize=(10, 6))
            plt.hist(all_ious, bins=50, edgecolor='black')
            plt.xlabel("IoU")
            plt.ylabel("Count")
            plt.title(f"IoU Distribution (Mean: {mean_iou:.3f})")
            plt.grid(True, alpha=0.3)
            plt.show()
    else:
        print("評価可能なデータがありませんでした")


def main():
    parser = argparse.ArgumentParser(description="SAM2.1モデルの評価と可視化")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="external/sam2/sam2_logs/foodmix_bplus_40ep/checkpoints/checkpoint.pt",
        help="学習済みチェックポイントのパス"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="external/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="モデル設定ファイルのパス"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/foodmix_sa1b",
        help="データディレクトリ"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="評価するデータスプリット"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="評価するサンプル数（Noneで全データ）"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="結果を可視化"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用デバイス"
    )
    
    args = parser.parse_args()
    
    # チェックポイント存在確認
    if not Path(args.checkpoint).exists():
        print(f"エラー: チェックポイントが見つかりません: {args.checkpoint}")
        print("学習を先に実行してください: bash scripts/20_train_food_sam2.sh")
        return
    
    # 評価実行
    evaluate_dataset(
        args.checkpoint,
        args.config,
        args.data_dir,
        args.split,
        args.num_samples,
        args.visualize,
        args.device
    )
    
    print("\n評価完了！")


if __name__ == "__main__":
    main()