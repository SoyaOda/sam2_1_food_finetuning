# SAM2.1食材学習データ前処理ガイド

このドキュメントでは、FoodSeg103とUEC-FoodPix Completeデータセットを、SAM2.1の学習に使用できるSA-1B形式に変換する手順を説明します。

## 概要

以下のデータセットをSA-1B互換形式（セマンティックマスク → インスタンスセグメンテーション）に変換します：

1. **UEC-FoodPix Complete** (10,000ファイル) - 食材レベルのセグメンテーション
2. **FoodSeg103** (7,118ファイル) - 食材カテゴリのセグメンテーション

**合計**: 17,118ペアのSA-1B形式データを生成

## 前提条件

### 環境セットアップ

```bash
# プロジェクトディレクトリで仮想環境を作成
uv venv

# 仮想環境をアクティベート
source .venv/bin/activate

# 必要なパッケージをインストール
uv pip install pycocotools pillow scikit-image tqdm numpy

# システムパッケージ（事前にインストール推奨）
sudo apt install -y python3-skimage python3-numpy python3-pil python3-scipy python3-matplotlib python3-tqdm
```

### データセット準備

1. **UEC-FoodPix Complete**: `data/UECFOODPIXCOMPLETE/` に配置
2. **FoodSeg103**: `data/FoodSeg103/` に配置

## Step 1: UEC-FoodPix Complete の前処理

### データセット構造
```
data/UECFOODPIXCOMPLETE/
├── data/UECFoodPIXCOMPLETE/
│   ├── train/
│   │   ├── img/           # JPG画像ファイル
│   │   └── mask/          # PNG マスクファイル（RチャネルにクラスID）
│   ├── test/
│   │   ├── img/
│   │   └── mask/
│   ├── train.txt          # 学習用ファイルリスト
│   ├── test.txt           # テスト用ファイルリスト
│   └── category.txt       # カテゴリ名一覧
```

### 実行方法

```bash
# 仮想環境で実行（必須）
source .venv/bin/activate

# UEC前処理スクリプトの実行
python scripts/11_prepare_uecfoodpix.py \
    --input data/UECFOODPIXCOMPLETE \
    --output data/foodmix_sa1b \
    --min-area 32
```

### パラメータ説明

- `--input`: UEC-FoodPix Completeデータセットのルートディレクトリ
- `--output`: SA-1B形式の出力ディレクトリ
- `--min-area`: インスタンスの最小サイズ（ピクセル数）。小さすぎるノイズを除去

## Step 2: FoodSeg103 の前処理

### データセット構造
```
data/FoodSeg103/
├── Images/
│   ├── train/           # 学習用JPG画像ファイル (4,983枚)
│   └── validation/      # 検証用JPG画像ファイル (2,135枚)
├── Masks/
│   ├── train/           # 学習用PNG マスクファイル
│   └── validation/      # 検証用PNG マスクファイル
└── dataset_info.txt     # データセット情報
```

### 実行方法

```bash
# 仮想環境で実行（必須）
source .venv/bin/activate

# 修正版FoodSeg103前処理の実行
python scripts/10_prepare_foodseg103.py \
    --input data/FoodSeg103 \
    --output data/foodmix_sa1b \
    --min-area 32
```

### パラメータ説明

- `--input`: FoodSeg103データセットのルートディレクトリ
- `--output`: SA-1B形式の出力ディレクトリ（UECと統合）
- `--min-area`: インスタンスの最小サイズ（ピクセル数、推奨: 32）

### 処理内容詳細

1. **セマンティック → インスタンス変換**:
   - FoodSeg103マスクの各クラスごとに連結成分分析（8連結）
   - `remove_small_objects`による事前ノイズ除去
   - 小領域インスタンス（< min_area）を自動除去

2. **SA-1B形式JSON生成**:
   - COCO RLE形式でマスクをエンコード（Fortran order対応）
   - バウンディングボックス、面積、中心点を自動計算
   - ユニークなimage_idを生成（ハッシュベース）

3. **エラーハンドリング**:
   - pycocotools環境依存問題に対応
   - 個別ファイルエラーで全体停止しない設計

## Step 3: データセット統合・スプリット作成（実装予定）

### 実行方法（予定）
```bash
# 複数データセットを統合してtrain/val/testスプリットを作成
python scripts/12_merge_to_sa1b.py \
    --input data/foodmix_sa1b \
    --output data/foodmix_sa1b_final \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

## 現在のデータ構成（前処理済み）

実際の前処理済みデータは以下の構造で保存されています：

```
data/foodmix_sa1b/
├── images/ (17,118ファイル)
│   ├── uec_train_*.jpg    (7,000枚) - UEC学習用
│   ├── uec_test_*.jpg     (3,000枚) - UECテスト用  
│   ├── train_train_*.jpg  (4,983枚) - FoodSeg103学習用
│   └── val_train_*.jpg    (2,135枚) - FoodSeg103検証用
├── annotations/ (17,118ファイル) - 対応するJSON
│   ├── uec_train_*.json   - UEC学習用アノテーション
│   ├── uec_test_*.json    - UECテスト用アノテーション
│   ├── train_train_*.json - FoodSeg103学習用アノテーション
│   └── val_train_*.json   - FoodSeg103検証用アノテーション
├── train.txt              (FoodSeg103用、4,983行)
├── val.txt                (FoodSeg103用、2,135行)
└── uec_files.txt          (UEC用、10,000行)
```

### データ確認コマンド

```bash
# 総ファイル数の確認
ls data/foodmix_sa1b/images/ | wc -l          # 17,118
ls data/foodmix_sa1b/annotations/ | wc -l     # 17,118

# UEC-FoodPix Completeデータの確認
find data/foodmix_sa1b/images/ -name "uec_*" | wc -l    # 10,000
wc -l data/foodmix_sa1b/uec_files.txt                   # 10,000

# FoodSeg103データの確認
find data/foodmix_sa1b/images/ -name "train_train_*" | wc -l  # 4,983
find data/foodmix_sa1b/images/ -name "val_train_*" | wc -l    # 2,135
wc -l data/foodmix_sa1b/train.txt                            # 4,983
wc -l data/foodmix_sa1b/val.txt                              # 2,135

# 画像とアノテーションの対応チェック
python -c "
import os
img_files = set(os.listdir('data/foodmix_sa1b/images/'))
ann_files = set(os.listdir('data/foodmix_sa1b/annotations/'))
matching = len([f for f in img_files if f.replace('.jpg', '.json') in ann_files])
print(f'Images: {len(img_files)}, Annotations: {len(ann_files)}, Matching: {matching}')
"
```

## SA-1B形式のデータ構造

### 画像とアノテーションの対応
- **画像**: `images/<basename>.jpg`
- **アノテーション**: `annotations/<basename>.json`

### JSON構造（SA-1B互換）
```json
{
  "image": {
    "image_id": 12345,
    "width": 1024,
    "height": 768,
    "file_name": "uec_train_12345.jpg"
  },
  "annotations": [
    {
      "id": 0,
      "segmentation": {
        "size": [768, 1024],
        "counts": "eNl02..."  // COCO RLE（Fortran order）
      },
      "bbox": [100.0, 50.0, 200.0, 150.0],  // [x, y, width, height]
      "area": 15420,
      "predicted_iou": 1.0,     // GT なので 1.0
      "stability_score": 1.0,   // GT なので 1.0
      "crop_box": [0.0, 0.0, 1024.0, 768.0],  // フル画像
      "point_coords": [[200, 125]],            // 中心点座標
      "category_id": 3          // 元のクラスID（参考用）
    }
  ]
}
```

## SAM2.1での使用方法

### Hydra設定例
```yaml
# configs/train_food_sa1b.yaml
defaults:
  - _self_

experiment_log_dir: ./sam2_logs/food_sa1b

phases_per_epoch: 1
num_train_workers: 8
max_num_objects_per_image: 5
bs1: 4     # SA-1B 画像側のバッチ

# データパス（絶対パス推奨）
path_to_img_folder: /abs/path/data/foodmix_sa1b/images
path_to_gt_folder:  /abs/path/data/foodmix_sa1b/annotations
path_to_train_filelist: /abs/path/data/foodmix_sa1b/train.txt   # FoodSeg103用

# 画像用 transform（公式実装に準拠）
image_transforms:
  _target_: training.dataset.transforms.ComposeAPI
  transforms:
    - _target_: training.dataset.transforms.ToTensorAPI
      v2: true
    - _target_: training.dataset.transforms.RandomResizeAPI
      sizes: [1024]
      consistent_transform: true
      square: true
      v2: true
    - _target_: training.dataset.transforms.RandomHorizontalFlip
      consistent_transform: true
      p: 0.5
    - _target_: training.dataset.transforms.ColorJitter
      consistent_transform: true
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    - _target_: training.dataset.transforms.RandomAffine
      degrees: 5
      consistent_transform: true
      scale: [0.9, 1.1]
      translate: null
      shear: 0
      image_mean: [123, 116, 103]
      image_interpolation: bilinear
      log_warning: true
      num_tentatives: 1
    - _target_: training.dataset.transforms.NormalizeAPI
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
      v2: true

data:
  train:
    _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset
    phases_per_epoch: ${phases_per_epoch}
    batch_sizes: [${bs1}]
    datasets:
      - _target_: training.dataset.vos_dataset.VOSDataset
        training: true
        video_dataset:
          _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
          img_folder: ${path_to_img_folder}
          gt_folder: ${path_to_gt_folder}
          file_list_txt: ${path_to_train_filelist}
          num_frames: 1
          mask_area_frac_thresh: 1.1   # 大きなマスクフィルタを無効化
          uncertain_iou: -1            # stabilityフィルタを無効化
        sampler:
          _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 1
        max_num_objects: ${max_num_objects_per_image}
        transforms: ${image_transforms}
    shuffle: true
    num_workers: ${num_train_workers}
    pin_memory: true
    drop_last: true
    collate_fn:
      _target_: training.utils.data_utils.collate_fn
      _partial_: true
      dict_key: all
```

### 学習実行例
```bash
# SAM2.1学習の実行
python training/train.py \
    -c configs/train_food_sa1b.yaml \
    --use-cluster 0 \
    --num-gpus 8
```

### ファイルリスト作成（カスタムスプリット用）

現在のデータを使って、カスタムファイルリストを作成できます：

```bash
# 全UECデータをtrainに使用
cp data/foodmix_sa1b/uec_files.txt data/foodmix_sa1b/all_train.txt

# FoodSeg103のtrainを追加
cat data/foodmix_sa1b/train.txt >> data/foodmix_sa1b/all_train.txt

# FoodSeg103のvalを検証用に使用
cp data/foodmix_sa1b/val.txt data/foodmix_sa1b/all_val.txt

# 合計確認
wc -l data/foodmix_sa1b/all_train.txt  # 14,983行（UEC 10,000 + FoodSeg103 train 4,983）
wc -l data/foodmix_sa1b/all_val.txt    # 2,135行（FoodSeg103 val）
```

## トラブルシューティング

### 一般的なエラーと対処法

#### 1. `ModuleNotFoundError: No module named 'PIL'`
```bash
# 仮想環境に必要なパッケージを再インストール
source .venv/bin/activate
uv pip install pillow scikit-image pycocotools tqdm numpy
```

#### 2. `list indices must be integers or slices, not str`
- pycocotoolsのバージョン問題（修正済み）
- 仮想環境を再作成してください

#### 3. `FileNotFoundError: データが見つかりません`
- データセットのディレクトリ構造を確認
- `--input`パラメータのパスが正しいかチェック

#### 4. 処理が途中で止まる
- メモリ不足の可能性
- `--min-area`を大きくしてインスタンス数を削減

### パフォーマンス最適化

#### メモリ使用量を削減
```bash
# 最小インスタンスサイズを大きく設定
python scripts/11_prepare_uecfoodpix.py \
    --min-area 64 \
    --input data/UECFOODPIXCOMPLETE \
    --output data/foodmix_sa1b
```

#### 処理速度を向上
- SSDストレージの使用
- 十分なRAM（8GB以上推奨）
- CPUコア数に応じたマルチプロセシング（将来の改善予定）

## 技術的詳細

### RLEエンコーディング
- **Fortran order**: pycocotoolsの要件（列優先）
- **Binary mask format**: `(H, W, 1)` uint8形式
- **JSON互換**: `counts`フィールドをASCII文字列に変換
- **環境依存対応**: pycocotoolsがリストを返す場合に対応

### 連結成分分析
- **アルゴリズム**: `skimage.measure.label`
- **連結性**: 8連結（connectivity=2）で品質向上
- **ノイズ除去**: `skimage.morphology.remove_small_objects`

### メモリ管理
- **遅延読み込み**: 大きなデータセットでもメモリ効率良く処理
- **エラー処理**: 個別ファイルのエラーで全体が停止しない設計
- **進行状況**: tqdmによるリアルタイム進行状況表示

## データ品質情報

### インスタンス統計
- **平均インスタンス数/画像**: 約5-15個（データセットにより変動）
- **最小インスタンスサイズ**: 32ピクセル（設定可能）
- **RLEエンコード効率**: 元マスクサイズの約10-30%に圧縮

### クラス分布
- **UEC-FoodPix**: 食材レベル（野菜、肉、魚介類など）
- **FoodSeg103**: 103カテゴリの食材分類

## 参考資料

- [SAM2.1公式トレーニングドキュメント](https://huggingface.co/spaces/3DAIGC/LHM/resolve/main/third_party/sam2/training/README.md)
- [SA-1B公式データ仕様](https://github.com/facebookresearch/segment-anything)
- [COCO RLE仕様](https://github.com/cocodataset/cocoapi)
- [pycocotools使用法](https://pycocotools.readthedocs.io/)
- [UEC-FoodPix Complete公式](https://mm.cs.uec.ac.jp/uecfoodpix/)
- [FoodSeg103論文](https://xiongweiwu.github.io/papers/FoodSeg103_ArXiv.pdf)

## 更新履歴

- 2024-09-06: UEC-FoodPix Complete前処理スクリプト完成（10,000ファイル処理確認済み）
- 2024-09-06: FoodSeg103前処理スクリプト修正完成（7,118ファイル、テスト確認済み）
- 2024-09-06: SA-1B形式互換性確認、pycocotools環境問題解決
- 2024-09-06: 合計17,118ペアのSA-1B形式データ完成