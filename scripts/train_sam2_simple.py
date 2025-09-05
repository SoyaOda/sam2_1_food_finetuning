#!/usr/bin/env python
"""
簡易トレーニングスクリプト - SAM2.1 for Food Segmentation
Hydra設定パスの問題を回避して直接学習を実行
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
from pathlib import Path

# SAM2モジュールのパスを追加
sys.path.insert(0, 'external/sam2')

# Hydraの初期化を先に行う
from hydra import initialize_config_module
initialize_config_module("sam2", version_base="1.2")

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

class FoodSA1BDataset(Dataset):
    """SA-1B形式のデータセット"""
    
    def __init__(self, img_dir, ann_dir, file_list, transform=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        
        # ファイルリストを読み込み
        if Path(file_list).exists():
            with open(file_list, 'r') as f:
                self.files = [line.strip() for line in f if line.strip()]
        else:
            # リストファイルがない場合は画像を直接取得
            self.files = [f.name for f in self.img_dir.glob('*.jpg')]
            self.files.extend([f.name for f in self.img_dir.glob('*.png')])
        
        self.transform = transform
        print(f"データセット初期化: {len(self.files)}個のファイル")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = self.img_dir / img_file
        
        # アノテーションファイル名を取得
        ann_name = img_file.rsplit('.', 1)[0] + '.json'
        ann_path = self.ann_dir / ann_name
        
        # 画像を読み込み
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # アノテーションを読み込み
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
                annotations = ann_data.get('annotations', [])
        else:
            annotations = []
        
        # マスクをデコード（簡易版：最初のマスクのみ）
        masks = []
        if annotations:
            from pycocotools import mask as mask_utils
            for ann in annotations[:5]:  # 最初の5つだけ処理
                rle = ann['segmentation']
                if isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode('ascii')
                mask = mask_utils.decode(rle)
                masks.append(mask)
        
        if not masks:
            # マスクがない場合はダミーマスクを作成
            h, w = image.shape[:2]
            masks = [np.zeros((h, w), dtype=np.uint8)]
        
        masks = np.stack(masks, axis=0)
        
        # Tensorに変換
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        masks = torch.from_numpy(masks).float()
        
        return {
            'image': image,
            'masks': masks,
            'file_name': img_file
        }

def train_simple():
    """簡易学習関数"""
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # チェックポイントが存在するか確認
    checkpoint_path = "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    if not os.path.exists(checkpoint_path):
        print(f"エラー: チェックポイントが見つかりません: {checkpoint_path}")
        print("'cd external/sam2/checkpoints && ./download_ckpts.sh' を実行してください")
        return
    
    # モデルの設定ファイル
    model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
    
    # SAM2モデルをビルド
    print("モデルをロード中...")
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
    
    # モデルを学習モードに
    sam2_model.train()
    
    # データセットの準備
    data_dir = Path("data/test_dataset")  # テスト用
    if not data_dir.exists():
        print(f"エラー: データディレクトリが見つかりません: {data_dir}")
        print("テストデータを先に作成してください")
        return
    
    img_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    
    # データローダーを作成
    train_dataset = FoodSA1BDataset(img_dir, ann_dir, data_dir / "train.txt")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # 簡易版なのでバッチサイズ1
        shuffle=True,
        num_workers=0
    )
    
    if len(train_dataset) == 0:
        print("エラー: データセットが空です")
        return
    
    # オプティマイザー
    optimizer = torch.optim.AdamW(sam2_model.parameters(), lr=1e-5)
    
    # 損失関数（簡易版：BCELoss）
    criterion = nn.BCEWithLogitsLoss()
    
    # 学習ループ
    num_epochs = 1  # テスト用に1エポックのみ
    print(f"\n学習開始: {num_epochs}エポック")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = batch['masks'].to(device)
            
            # 順伝播（簡易版）
            # 注意: 実際のSAM2学習は複雑なので、これは概念実証のみ
            with torch.no_grad():
                # エンコーダーの特徴を取得
                backbone_out = sam2_model.forward_image(images)
                
            # ここで実際の学習処理を行う（詳細は省略）
            # 簡易的な損失を計算
            dummy_loss = torch.tensor(0.1, requires_grad=True)
            
            # 逆伝播
            optimizer.zero_grad()
            dummy_loss.backward()
            optimizer.step()
            
            epoch_loss += dummy_loss.item()
            progress_bar.set_postfix(loss=dummy_loss.item())
            
            # テスト用：最初の5バッチで終了
            if batch_idx >= 4:
                break
        
        avg_loss = epoch_loss / min(len(train_loader), 5)
        print(f"Epoch {epoch+1} - 平均損失: {avg_loss:.4f}")
    
    print("\n学習完了（テスト実行）")
    print("実際の学習にはより詳細な実装が必要です")

if __name__ == "__main__":
    train_simple()