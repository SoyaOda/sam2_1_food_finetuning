# SAM2.1 Food Finetuning - RandomBlurAPI Error

## 現状の実装状況

### 完了したタスク
1. FoodSeg103データセットをダウンロード済み（4,983枚の訓練画像、2,135枚の検証画像）
2. SA-1B形式への変換スクリプト作成済み（`scripts/10_prepare_foodseg103.py`）
3. Hydra設定パス問題は`scripts/train_sam2_wrapper.py`のラッパースクリプトで解決
4. `checkpoint_block_num`パラメータエラーは設定ファイルから削除して修正済み

### 現在のエラー
```
ModuleNotFoundError: No module named 'training.dataset.transforms.RandomBlurAPI'; 'training.dataset.transforms' is not a package
```

### エラーの詳細
- 設定ファイル: `external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_40epochs.yaml`
- 問題箇所: `vos.train_transforms`内の`RandomBlurAPI`トランスフォーム
- 実際のtransforms.pyに存在するクラス:
  - RandomHorizontalFlip
  - RandomResizeAPI
  - ToTensorAPI
  - NormalizeAPI
  - ComposeAPI
  - RandomGrayscale
  - ColorJitter
  - RandomAffine
  - RandomMosaicVideoAPI
  （RandomBlurAPIは存在しない）

### 設定ファイルの該当箇所
```yaml
vos:
  train_transforms:
    - _target_: training.dataset.transforms.ComposeAPI
      transforms:
        # ... 他のトランスフォーム ...
        - _target_: training.dataset.transforms.RandomBlurAPI  # このクラスが存在しない
          prob: 0.25
          sim_blur_prob: 0.5
          consistent_transform: True
          kernel_sizes: [3, 5, 7, 9]
```

### ファイル構成
- 学習スクリプト: `external/sam2/training/train.py`
- ラッパー: `scripts/train_sam2_wrapper.py`
- 設定: `external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_40epochs.yaml`
- データ: `/home/soya/sam2_1_food_finetuning/data/foodmix_sa1b/`

### モデル情報
- SAM2.1 Hiera Base+ (80.9M parameters)
- チェックポイント: `external/sam2/checkpoints/sam2.1_hiera_base_plus.pt`