#!/usr/bin/env python3
"""
SAM2.1 Hiera-Large 予測結果の可視化スクリプト
学習済みLargeモデルの定性評価を行う
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
            logging.FileHandler('visualization_large.log')
        ]
    )

def load_sam2_large_model(checkpoint_path: str, model_cfg: str = "sam2_hiera_l.yaml") -> Optional[SAM2ImagePredictor]:
    """
    SAM2 Largeモデルを読み込み
    """
    if not SAM2_AVAILABLE:
        logging.error("SAM2が利用できません")
        return None
    
    try:
        # Largeモデル用設定パス - configs/sam2.1/ を参照
        model_cfg_path = os.path.join(sam2_path, "sam2", "configs", "sam2.1", model_cfg)
        
        if not os.path.exists(checkpoint_path):
            logging.error(f"チェックポイントが見つかりません: {checkpoint_path}")
            return None
        
        if not os.path.exists(model_cfg_path):
            logging.error(f"モデル設定が見つかりません: {model_cfg_path}")
            return None
        
        # SAM2 Largeモデル構築
        logging.info(f"Loading SAM2.1 Hiera-Large model...")
        logging.info(f"Config: {model_cfg_path}")
        logging.info(f"Checkpoint: {checkpoint_path}")
        
        sam2_model = build_sam2(model_cfg_path, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        
        logging.info(f"SAM2.1 Hiera-Large モデル読み込み完了")
        
        # モデル情報表示
        total_params = sum(p.numel() for p in sam2_model.parameters())
        trainable_params = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        return predictor
        
    except Exception as e:
        logging.error(f"SAM2 Largeモデル読み込み失敗: {e}")
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
        
        logging.debug(f"グラウンドトゥルース読み込み: {len(masks)} マスク")
        return masks
        
    except Exception as e:
        logging.error(f"アノテーション読み込み失敗 {annotation_path}: {e}")
        return []

def predict_masks_large(predictor: SAM2ImagePredictor, image: np.ndarray, 
                        num_masks: int = 5) -> Tuple[List[np.ndarray], List[float]]:
    """
    SAM2 Largeモデルで自動マスク予測を実行
    """
    try:
        logging.debug(f"Setting image for Large model prediction...")
        start_time = time.time() if 'time' in dir() else None
        
        predictor.set_image(image)
        
        # Largeモデル用：より精密な点プロンプト設定
        h, w = image.shape[:2]
        
        # 複数の点を使用してより良い予測を得る
        points = [
            [w//2, h//2],           # 中央
            [w//4, h//4],           # 左上
            [3*w//4, h//4],         # 右上
            [w//4, 3*h//4],         # 左下
            [3*w//4, 3*h//4],       # 右下
        ]
        
        all_masks = []
        all_scores = []
        
        # 各点に対して予測実行
        for point in points:
            point_coords = np.array([point])
            point_labels = np.array([1])
            
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            # 結果を収集
            for mask, score in zip(masks, scores):
                all_masks.append(mask)
                all_scores.append(score)
        
        # スコア順でソートして上位を選択
        sorted_indices = np.argsort(all_scores)[::-1]
        
        # 重複マスクを除去（簡易版）
        unique_masks = []
        unique_scores = []
        
        for idx in sorted_indices:
            mask = all_masks[idx]
            score = all_scores[idx]
            
            # 既存マスクとのIoUをチェック
            is_duplicate = False
            for existing_mask in unique_masks:
                intersection = np.logical_and(mask, existing_mask).sum()
                union = np.logical_or(mask, existing_mask).sum()
                if union > 0:
                    iou = intersection / union
                    if iou > 0.5:  # 50%以上重複している場合は除外
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_masks.append(mask)
                unique_scores.append(score)
                
                if len(unique_masks) >= num_masks:
                    break
        
        end_time = time.time() if start_time else None
        if end_time:
            logging.debug(f"Large model prediction time: {end_time - start_time:.2f}s")
        
        logging.info(f"予測完了: {len(unique_masks)} unique masks from Large model")
        return unique_masks, unique_scores
        
    except Exception as e:
        logging.error(f"Large model prediction failed: {e}")
        return [], []

# 時間計測用のimport
import time

def visualize_comparison_large(image: np.ndarray, gt_masks: List[np.ndarray], 
                               pred_masks: List[np.ndarray], pred_scores: List[float],
                               output_path: str, image_name: str):
    """
    Largeモデル用：グラウンドトゥルースと予測結果の比較可視化
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Largeモデル用に拡大
    fig.suptitle(f'SAM2.1 Hiera-Large Prediction Comparison: {image_name}', fontsize=16)
    
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
    
    # IoU計算（詳細版 - Largeモデル用）
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
    for i, (mask, score) in enumerate(zip(pred_masks[:3], pred_scores[:3])):
        color = (colors_pred[i][:3] * 255).astype(np.uint8)
        pred_overlay[mask > 0] = pred_overlay[mask > 0] * 0.6 + color * 0.4
    
    axes[0, 2].imshow(cv2.cvtColor(pred_overlay.astype(np.uint8), cv2.COLOR_BGR_RGB))
    pred_title = f'Large Model Predictions (top 3)'
    if iou_scores:
        avg_iou = np.mean(iou_scores[:3])
        pred_title += f'\nAvg IoU: {avg_iou:.3f}'
    axes[0, 2].set_title(pred_title)
    axes[0, 2].axis('off')
    
    # 個別予測マスク（トップ3）
    for i in range(3):
        ax = axes[1, i]
        if i < len(pred_masks):
            ax.imshow(pred_masks[i], cmap='gray')
            title = f'Large Pred {i+1} (score: {pred_scores[i]:.3f})'
            if i < len(iou_scores):
                title += f'\nIoU: {iou_scores[i]:.3f}'
                # IoUに基づく品質評価
                if iou_scores[i] > 0.7:
                    title += ' ✓'
                elif iou_scores[i] > 0.5:
                    title += ' ○'
                elif iou_scores[i] > 0.3:
                    title += ' △'
                else:
                    title += ' ×'
            ax.set_title(title)
        else:
            ax.set_title('No prediction')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Large model visualization saved: {output_path}")
    
    return iou_scores

def process_image_list_large(predictor: SAM2ImagePredictor, image_list_file: str, 
                            img_folder: str, gt_folder: str, output_dir: str, 
                            max_images: int = 10) -> dict:
    """
    Largeモデル用：画像リストを処理して予測結果を可視化
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像リスト読み込み
    with open(image_list_file, 'r') as f:
        image_files = [line.strip() for line in f if line.strip()]
    
    # 最大処理数を制限
    image_files = image_files[:max_images]
    
    results = {
        'processed_images': [],
        'iou_scores': [],
        'prediction_scores': [],
        'processing_times': [],
        'model_type': 'SAM2.1 Hiera-Large'
    }
    
    logging.info(f"Processing {len(image_files)} images with Large model...")
    
    for i, image_file in enumerate(image_files):
        try:
            start_time = time.time()
            
            # 画像読み込み
            image_path = os.path.join(img_folder, f"{image_file}.jpg")
            if not os.path.exists(image_path):
                logging.warning(f"Image not found: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue
            
            # アノテーション読み込み
            annotation_path = os.path.join(gt_folder, f"{image_file}.json")
            gt_masks = load_ground_truth_masks(annotation_path)
            
            # Large モデル予測実行
            pred_masks, pred_scores = predict_masks_large(predictor, image)
            
            if not pred_masks:
                logging.warning(f"No predictions for: {image_file}")
                continue
            
            # 可視化
            output_path = os.path.join(output_dir, f"{image_file}_large_comparison.png")
            iou_scores = visualize_comparison_large(
                image, gt_masks, pred_masks, pred_scores, output_path, image_file
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 結果記録
            results['processed_images'].append(image_file)
            results['iou_scores'].append(iou_scores)
            results['prediction_scores'].append(pred_scores)
            results['processing_times'].append(processing_time)
            
            logging.info(f"Processed ({i+1}/{len(image_files)}): {image_file} (time: {processing_time:.2f}s)")
            
        except Exception as e:
            logging.error(f"Processing error {image_file}: {e}")
            continue
    
    return results

def analyze_large_results(results: dict, output_dir: str):
    """
    Largeモデル予測結果の詳細統計分析
    """
    logging.info("=== Large Model Prediction Analysis ===")
    
    total_processed = len(results['processed_images'])
    logging.info(f"Processed images: {total_processed}")
    
    if total_processed == 0:
        logging.warning("No results to analyze")
        return
    
    # IoU統計
    all_ious = []
    for image_ious in results['iou_scores']:
        if image_ious:
            all_ious.extend(image_ious[:3])  # トップ3のみ
    
    if all_ious:
        avg_iou = np.mean(all_ious)
        std_iou = np.std(all_ious)
        max_iou = np.max(all_ious)
        min_iou = np.min(all_ious)
        
        logging.info(f"IoU Statistics:")
        logging.info(f"  Average: {avg_iou:.4f}")
        logging.info(f"  Std Dev: {std_iou:.4f}")
        logging.info(f"  Max: {max_iou:.4f}")
        logging.info(f"  Min: {min_iou:.4f}")
        
        # 品質分類
        high_quality = [iou for iou in all_ious if iou > 0.7]
        good_quality = [iou for iou in all_ious if 0.5 < iou <= 0.7]
        fair_quality = [iou for iou in all_ious if 0.3 < iou <= 0.5]
        poor_quality = [iou for iou in all_ious if iou <= 0.3]
        
        logging.info(f"Quality Distribution:")
        logging.info(f"  Excellent (>0.7): {len(high_quality)} ({len(high_quality)/len(all_ious)*100:.1f}%)")
        logging.info(f"  Good (0.5-0.7): {len(good_quality)} ({len(good_quality)/len(all_ious)*100:.1f}%)")
        logging.info(f"  Fair (0.3-0.5): {len(fair_quality)} ({len(fair_quality)/len(all_ious)*100:.1f}%)")
        logging.info(f"  Poor (≤0.3): {len(poor_quality)} ({len(poor_quality)/len(all_ious)*100:.1f}%)")
    
    # 処理時間統計
    if results['processing_times']:
        avg_time = np.mean(results['processing_times'])
        total_time = sum(results['processing_times'])
        
        logging.info(f"Processing Time Statistics:")
        logging.info(f"  Average per image: {avg_time:.2f}s")
        logging.info(f"  Total time: {total_time:.2f}s")
        logging.info(f"  Estimated 40-epoch training time: {avg_time * 100 * 40 / 3600:.1f} hours")
    
    # 詳細レポート作成
    summary_path = os.path.join(output_dir, 'large_model_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SAM2.1 Hiera-Large Model Prediction Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {results['model_type']}\n")
        f.write(f"Processed Images: {total_processed}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if all_ious:
            f.write(f"IoU Performance:\n")
            f.write(f"  Average: {avg_iou:.4f} ± {std_iou:.4f}\n")
            f.write(f"  Range: {min_iou:.4f} - {max_iou:.4f}\n\n")
            
            f.write(f"Quality Breakdown:\n")
            f.write(f"  Excellent (>0.7): {len(high_quality)}/{len(all_ious)} ({len(high_quality)/len(all_ious)*100:.1f}%)\n")
            f.write(f"  Good (0.5-0.7): {len(good_quality)}/{len(all_ious)} ({len(good_quality)/len(all_ious)*100:.1f}%)\n")
            f.write(f"  Fair (0.3-0.5): {len(fair_quality)}/{len(all_ious)} ({len(fair_quality)/len(all_ious)*100:.1f}%)\n")
            f.write(f"  Poor (≤0.3): {len(poor_quality)}/{len(all_ious)} ({len(poor_quality)/len(all_ious)*100:.1f}%)\n\n")
        
        if results['processing_times']:
            f.write(f"Performance:\n")
            f.write(f"  Avg processing time: {avg_time:.2f}s per image\n")
            f.write(f"  Total processing time: {total_time:.2f}s\n\n")
        
        f.write("Recommendations:\n")
        if all_ious:
            if avg_iou > 0.6:
                f.write("✓ Excellent: Large model shows strong performance\n")
            elif avg_iou > 0.5:
                f.write("○ Good: Large model performance is satisfactory\n")
            elif avg_iou > 0.3:
                f.write("△ Fair: Consider additional fine-tuning\n")
            else:
                f.write("× Poor: Review training data and hyperparameters\n")
    
    logging.info(f"Detailed analysis saved: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="SAM2.1 Hiera-Large Prediction Visualization")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="SAM2 Large model checkpoint path")
    parser.add_argument("--image-list", type=str, required=True,
                       help="Image list file path")
    parser.add_argument("--img-folder", type=str, required=True,
                       help="Image folder path")
    parser.add_argument("--gt-folder", type=str, required=True,
                       help="Ground truth annotation folder path")
    parser.add_argument("--output-dir", type=str, default="visualization_results_large",
                       help="Output directory")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to process")
    parser.add_argument("--model-cfg", type=str, default="sam2.1_hiera_l.yaml",
                       help="SAM2 Large model config file")
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("SAM2.1 Hiera-Large Model Prediction Visualization")
    logging.info("=" * 60)
    logging.info(f"Model config: {args.model_cfg}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Max images: {args.max_images}")
    
    # SAM2 Large モデル読み込み
    predictor = load_sam2_large_model(args.checkpoint, args.model_cfg)
    if predictor is None:
        logging.error("Failed to load SAM2 Large model")
        return 1
    
    # 予測・可視化実行
    results = process_image_list_large(
        predictor, 
        args.image_list, 
        args.img_folder, 
        args.gt_folder, 
        args.output_dir, 
        args.max_images
    )
    
    # 結果分析
    analyze_large_results(results, args.output_dir)
    
    logging.info("=" * 60)
    logging.info("Large Model Visualization Complete")
    logging.info("=" * 60)
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info("Next steps:")
    logging.info("1. Review visualization images for prediction quality")
    logging.info("2. Compare IoU distribution with Base+ model results")
    logging.info("3. Identify problematic images for further analysis")
    logging.info("4. Consider parameter tuning if performance is suboptimal")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())