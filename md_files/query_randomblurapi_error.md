# Query: SAM2.1のRandomBlurAPIエラーの解決方法

## 背景
SAM2.1を食品画像セグメンテーション（FoodSeg103データセット）用にファインチューニングしようとしています。

## エラー内容
```python
ModuleNotFoundError: No module named 'training.dataset.transforms.RandomBlurAPI'; 'training.dataset.transforms' is not a package
```

## 現在の実装状況

### 1. 存在するトランスフォームクラス（external/sam2/training/dataset/transforms.py）
- RandomHorizontalFlip
- RandomResizeAPI
- ToTensorAPI
- NormalizeAPI
- ComposeAPI
- RandomGrayscale
- ColorJitter
- RandomAffine
- RandomMosaicVideoAPI

RandomBlurAPIクラスは存在しません。

### 2. 設定ファイルの該当箇所（sam2.1_hiera_b+_foodmix_40epochs.yaml）
```yaml
vos:
  train_transforms:
    - _target_: training.dataset.transforms.ComposeAPI
      transforms:
        - _target_: training.dataset.transforms.RandomHorizontalFlip
          consistent_transform: True
        - _target_: training.dataset.transforms.RandomAffine
          degrees: 25
          shear: 20
          image_interpolation: bilinear
          consistent_transform: True
        - _target_: training.dataset.transforms.RandomResizeAPI
          sizes: ${scratch.resolution}
          square: true
          consistent_transform: True
        - _target_: training.dataset.transforms.ColorJitter
          consistent_transform: True
          brightness: 0.1
          contrast: 0.03
          saturation: 0.03
          hue: null
        - _target_: training.dataset.transforms.RandomGrayscale
          p: 0.05
          consistent_transform: True
        - _target_: training.dataset.transforms.RandomBlurAPI  # ← このクラスが存在しない
          prob: 0.25
          sim_blur_prob: 0.5
          consistent_transform: True
          kernel_sizes: [3, 5, 7, 9]
        - _target_: training.dataset.transforms.ToTensorAPI
        - _target_: training.dataset.transforms.NormalizeAPI
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
```

## 質問

1. **SAM2.1の公式実装におけるRandomBlurの正しい実装方法は？**
   - RandomBlurAPIの代替となる正しいクラス名は何か？
   - torchvision.transforms.GaussianBlurを使うべきか、カスタム実装が必要か？

2. **SAM2.1のtraining.dataset.transformsモジュールでBlur augmentationを適用する正しい方法は？**
   - 公式のSAM2.1実装例での画像ブラー処理の実装方法
   - consistent_transformパラメータとの互換性の保ち方

3. **食品画像セグメンテーション用のData Augmentationとして推奨される設定は？**
   - SAM2.1で食品画像に特化したaugmentationの推奨設定
   - RandomBlurを削除しても問題ないか、それとも代替実装が必要か？

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。