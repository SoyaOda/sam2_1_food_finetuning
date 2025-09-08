#!/usr/bin/env python3
"""
SAM2.1 予測結果の可視化スクリプト
学習済みモデルの定性評価を行う
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import torch

# SAM2のパスを追加
sam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "sam2"))
sys.path.insert(0, sam2_path)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import pycocotools.mask as mask_util
    SAM2_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SAM2 import failed: {e}")
    SAM2_AVAILABLE = False

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('visualization.log')
        ]
    )

def load_sam2_model(checkpoint_path: str, model_cfg: str = "sam2_hiera_b+.yaml") -> Optional[SAM2ImagePredictor]:
    """
    SAM2モデルを読み込み
    """
    if not SAM2_AVAILABLE:
        logging.error("SAM2が利用できません")
        return None
    
    try:
        # モデル設定パス
        model_cfg_path = os.path.join(sam2_path, "sam2_configs", model_cfg)
        
        if not os.path.exists(checkpoint_path):
            logging.error(f"チェックポイントが見つかりません: {checkpoint_path}")
            return None
        
        if not os.path.exists(model_cfg_path):
            logging.error(f"モデル設定が見つかりません: {model_cfg_path}")
            return None
        
        # SAM2モデル構築
        sam2_model = build_sam2(model_cfg_path, checkpoint_path)
        predictor = SAM2ImagePredictor(sam2_model)
        
        logging.info(f"SAM2モデル読み込み完了: {checkpoint_path}")
        return predictor
        
    except Exception as e:
        logging.error(f"SAM2モデル読み込み失敗: {e}")
        return None

def load_ground_truth_masks(annotation_path: str) -> List[np.ndarray]:
    """
    SA-1B形式のアノテーションからグラウンドトゥルースマスクを読み込み
    """
    try:
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        masks = []
        for ann in annotation.get('annotations', []):
            if 'segmentation' in ann:
                # RLE形式をデコード
                rle = ann['segmentation']
                mask = mask_util.decode(rle)
                masks.append(mask)
        
        logging.info(f"グラウンドトゥルース読み込み: {len(masks)} マスク")
        return masks
        
    except Exception as e:
        logging.error(f"アノテーション読み込み失敗 {annotation_path}: {e}")
        return []

def predict_masks(predictor: SAM2ImagePredictor, image: np.ndarray, 
                  num_masks: int = 5) -> List[np.ndarray]:
    """
    SAM2で自動マスク予測を実行
    """
    try:
        predictor.set_image(image)
        
        # 自動マスク生成（点プロンプト無しで全体予測）
        # 画像中心を点プロンプトとして使用
        h, w = image.shape[:2]
        center_point = np.array([[w//2, h//2]])
        center_label = np.array([1])
        
        masks, scores, logits = predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True
        )
        
        # スコア順でソート
        sorted_indices = np.argsort(scores)[::-1]
        top_masks = [masks[i] for i in sorted_indices[:num_masks]]
        top_scores = [scores[i] for i in sorted_indices[:num_masks]]
        
        logging.info(f"予測完了: {len(top_masks)} マスク")
        return top_masks, top_scores
        
    except Exception as e:
        logging.error(f"予測失敗: {e}")
        return [], []

def visualize_comparison(image: np.ndarray, gt_masks: List[np.ndarray], 
                        pred_masks: List[np.ndarray], pred_scores: List[float],
                        output_path: str, image_name: str):
    """
    グラウンドトゥルースと予測結果の比較可視化
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'SAM2 Prediction Comparison: {image_name}', fontsize=16)
    
    # 元画像
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # グラウンドトゥルース（最大5個まで）
    gt_overlay = image.copy()
    colors_gt = plt.cm.Set1(np.linspace(0, 1, min(5, len(gt_masks))))
    for i, mask in enumerate(gt_masks[:5]):
        color = (colors_gt[i][:3] * 255).astype(np.uint8)
        gt_overlay[mask > 0] = gt_overlay[mask > 0] * 0.6 + color * 0.4
    
    axes[0, 1].imshow(cv2.cvtColor(gt_overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Ground Truth ({len(gt_masks)} masks)')
    axes[0, 1].axis('off')
    
    # IoU計算（簡易版）
    iou_scores = []
    if gt_masks and pred_masks:
        for pred_mask in pred_masks:
            best_iou = 0
            for gt_mask in gt_masks:
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union > 0:
                    iou = intersection / union
                    best_iou = max(best_iou, iou)
            iou_scores.append(best_iou)
    
    # 予測結果（トップ3）
    pred_overlay = image.copy()
    colors_pred = plt.cm.Set2(np.linspace(0, 1, min(3, len(pred_masks))))
    for i, (mask, score) in enumerate(zip(pred_masks[:3], pred_scores[:3])):\n        color = (colors_pred[i][:3] * 255).astype(np.uint8)\n        pred_overlay[mask > 0] = pred_overlay[mask > 0] * 0.6 + color * 0.4\n    \n    axes[0, 2].imshow(cv2.cvtColor(pred_overlay.astype(np.uint8), cv2.COLOR_BGR_RGB))\n    pred_title = f'Predictions (top 3)'\n    if iou_scores:\n        avg_iou = np.mean(iou_scores[:3])\n        pred_title += f'\\nAvg IoU: {avg_iou:.3f}'\n    axes[0, 2].set_title(pred_title)\n    axes[0, 2].axis('off')\n    \n    # 個別予測マスク（トップ3）\n    for i in range(3):\n        ax = axes[1, i]\n        if i < len(pred_masks):\n            ax.imshow(pred_masks[i], cmap='gray')\n            title = f'Pred {i+1} (score: {pred_scores[i]:.3f})'\n            if i < len(iou_scores):\n                title += f'\\nIoU: {iou_scores[i]:.3f}'\n            ax.set_title(title)\n        else:\n            ax.set_title('No prediction')\n        ax.axis('off')\n    \n    plt.tight_layout()\n    plt.savefig(output_path, dpi=150, bbox_inches='tight')\n    plt.close()\n    \n    logging.info(f\"可視化保存: {output_path}\")\n    \n    return iou_scores\n\ndef process_image_list(predictor: SAM2ImagePredictor, image_list_file: str, \n                      img_folder: str, gt_folder: str, output_dir: str, \n                      max_images: int = 10) -> dict:\n    \"\"\"\n    画像リストを処理して予測結果を可視化\n    \"\"\"\n    os.makedirs(output_dir, exist_ok=True)\n    \n    # 画像リスト読み込み\n    with open(image_list_file, 'r') as f:\n        image_files = [line.strip() for line in f if line.strip()]\n    \n    # 最大処理数を制限\n    image_files = image_files[:max_images]\n    \n    results = {\n        'processed_images': [],\n        'iou_scores': [],\n        'prediction_scores': []\n    }\n    \n    logging.info(f\"処理開始: {len(image_files)} 画像\")\n    \n    for i, image_file in enumerate(image_files):\n        try:\n            # 画像読み込み\n            image_path = os.path.join(img_folder, f\"{image_file}.jpg\")\n            if not os.path.exists(image_path):\n                logging.warning(f\"画像が見つかりません: {image_path}\")\n                continue\n            \n            image = cv2.imread(image_path)\n            if image is None:\n                logging.warning(f\"画像読み込み失敗: {image_path}\")\n                continue\n            \n            # アノテーション読み込み\n            annotation_path = os.path.join(gt_folder, f\"{image_file}.json\")\n            gt_masks = load_ground_truth_masks(annotation_path)\n            \n            # 予測実行\n            pred_masks, pred_scores = predict_masks(predictor, image)\n            \n            if not pred_masks:\n                logging.warning(f\"予測失敗: {image_file}\")\n                continue\n            \n            # 可視化\n            output_path = os.path.join(output_dir, f\"{image_file}_comparison.png\")\n            iou_scores = visualize_comparison(\n                image, gt_masks, pred_masks, pred_scores, output_path, image_file\n            )\n            \n            # 結果記録\n            results['processed_images'].append(image_file)\n            results['iou_scores'].append(iou_scores)\n            results['prediction_scores'].append(pred_scores)\n            \n            logging.info(f\"処理完了 ({i+1}/{len(image_files)}): {image_file}\")\n            \n        except Exception as e:\n            logging.error(f\"処理エラー {image_file}: {e}\")\n            continue\n    \n    return results\n\ndef analyze_results(results: dict, output_dir: str):\n    \"\"\"\n    予測結果の統計分析\n    \"\"\"\n    logging.info(\"=== 予測結果分析 ===\")\n    \n    total_processed = len(results['processed_images'])\n    logging.info(f\"処理画像数: {total_processed}\")\n    \n    if total_processed == 0:\n        logging.warning(\"分析する結果がありません\")\n        return\n    \n    # IoU統計\n    all_ious = []\n    for image_ious in results['iou_scores']:\n        if image_ious:\n            all_ious.extend(image_ious[:3])  # トップ3のみ\n    \n    if all_ious:\n        avg_iou = np.mean(all_ious)\n        std_iou = np.std(all_ious)\n        max_iou = np.max(all_ious)\n        min_iou = np.min(all_ious)\n        \n        logging.info(f\"IoU統計:\")\n        logging.info(f\"  平均: {avg_iou:.4f}\")\n        logging.info(f\"  標準偏差: {std_iou:.4f}\")\n        logging.info(f\"  最大: {max_iou:.4f}\")\n        logging.info(f\"  最小: {min_iou:.4f}\")\n        \n        # IoU分布のヒストグラム\n        plt.figure(figsize=(10, 6))\n        plt.hist(all_ious, bins=20, alpha=0.7, edgecolor='black')\n        plt.xlabel('IoU Score')\n        plt.ylabel('Frequency')\n        plt.title('IoU Score Distribution')\n        plt.axvline(avg_iou, color='red', linestyle='--', label=f'Mean: {avg_iou:.3f}')\n        plt.legend()\n        plt.grid(True, alpha=0.3)\n        \n        hist_path = os.path.join(output_dir, 'iou_distribution.png')\n        plt.savefig(hist_path, dpi=150, bbox_inches='tight')\n        plt.close()\n        \n        logging.info(f\"IoU分布グラフ保存: {hist_path}\")\n    \n    # 予測スコア統計\n    all_pred_scores = []\n    for image_scores in results['prediction_scores']:\n        if image_scores:\n            all_pred_scores.extend(image_scores[:3])  # トップ3のみ\n    \n    if all_pred_scores:\n        avg_pred_score = np.mean(all_pred_scores)\n        logging.info(f\"予測スコア統計:\")\n        logging.info(f\"  平均: {avg_pred_score:.4f}\")\n        logging.info(f\"  最大: {np.max(all_pred_scores):.4f}\")\n        logging.info(f\"  最小: {np.min(all_pred_scores):.4f}\")\n    \n    # サマリーレポート作成\n    summary_path = os.path.join(output_dir, 'analysis_summary.txt')\n    with open(summary_path, 'w') as f:\n        f.write(\"SAM2 Prediction Analysis Summary\\n\")\n        f.write(\"=\" * 40 + \"\\n\")\n        f.write(f\"Processed Images: {total_processed}\\n\")\n        if all_ious:\n            f.write(f\"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}\\n\")\n            f.write(f\"IoU Range: {min_iou:.4f} - {max_iou:.4f}\\n\")\n        if all_pred_scores:\n            f.write(f\"Average Prediction Score: {avg_pred_score:.4f}\\n\")\n        f.write(\"\\n\")\n        f.write(\"Quality Assessment:\\n\")\n        if all_ious:\n            if avg_iou > 0.5:\n                f.write(\"✓ Good: Average IoU > 0.5\\n\")\n            elif avg_iou > 0.3:\n                f.write(\"⚠ Fair: Average IoU 0.3-0.5\\n\")\n            else:\n                f.write(\"✗ Poor: Average IoU < 0.3\\n\")\n    \n    logging.info(f\"分析サマリー保存: {summary_path}\")\n\ndef main():\n    parser = argparse.ArgumentParser(description=\"SAM2.1 予測結果可視化\")\n    parser.add_argument(\"--checkpoint\", type=str, required=True,\n                       help=\"SAM2チェックポイントパス\")\n    parser.add_argument(\"--image-list\", type=str, required=True,\n                       help=\"画像リストファイル\")\n    parser.add_argument(\"--img-folder\", type=str, required=True,\n                       help=\"画像フォルダ\")\n    parser.add_argument(\"--gt-folder\", type=str, required=True,\n                       help=\"アノテーションフォルダ\")\n    parser.add_argument(\"--output-dir\", type=str, default=\"visualization_results\",\n                       help=\"出力ディレクトリ\")\n    parser.add_argument(\"--max-images\", type=int, default=10,\n                       help=\"最大処理画像数\")\n    parser.add_argument(\"--model-cfg\", type=str, default=\"sam2_hiera_b+.yaml\",\n                       help=\"SAM2モデル設定ファイル\")\n    \n    args = parser.parse_args()\n    \n    setup_logging()\n    \n    logging.info(\"=\" * 60)\n    logging.info(\"SAM2.1 予測結果可視化開始\")\n    logging.info(\"=\" * 60)\n    \n    # SAM2モデル読み込み\n    predictor = load_sam2_model(args.checkpoint, args.model_cfg)\n    if predictor is None:\n        logging.error(\"SAM2モデル読み込み失敗\")\n        return 1\n    \n    # 予測・可視化実行\n    results = process_image_list(\n        predictor, \n        args.image_list, \n        args.img_folder, \n        args.gt_folder, \n        args.output_dir, \n        args.max_images\n    )\n    \n    # 結果分析\n    analyze_results(results, args.output_dir)\n    \n    logging.info(\"=\" * 60)\n    logging.info(\"可視化完了\")\n    logging.info(\"=\" * 60)\n    logging.info(f\"結果: {args.output_dir}\")\n    logging.info(\"次のステップ:\")\n    logging.info(\"1. 可視化画像で予測品質を確認\")\n    logging.info(\"2. IoU分布で定量評価\")\n    logging.info(\"3. 問題のある画像を特定して設定調整\")\n    \n    return 0\n\nif __name__ == \"__main__\":\n    sys.exit(main())