# SAM2.1 Hydra設定パス問題の解決方法

## 現在の問題状況

SAM2.1の学習スクリプト実行時に、Hydra設定ファイルが見つからないエラーが発生しています。

### エラー詳細
```
hydra.errors.MissingConfigException: Cannot find primary config 'sam2.1_training/sam2.1_hiera_b+_foodmix_finetune'. 
Check that it's in your config search path.

Config search path:
	provider=hydra, path=pkg://hydra.conf
	provider=main, path=pkg://sam2
	provider=schema, path=structured://
```

### 現在のディレクトリ構造
```
/home/soya/sam2_1_food_finetuning/
├── external/
│   └── sam2/                          # Meta公式SAM2リポジトリ
│       ├── sam2/
│       │   └── configs/
│       │       └── sam2.1_training/
│       │           ├── sam2.1_hiera_b+_MOSE_finetune.yaml  # 公式の設定（動作しない）
│       │           ├── sam2.1_hiera_b+_foodmix_finetune.yaml  # カスタム設定（見つからない）
│       │           └── sam2.1_hiera_b+_foodmix_test.yaml  # テスト設定（見つからない）
│       ├── training/
│       │   └── train.py               # メインの学習スクリプト
│       └── checkpoints/
│           └── sam2.1_hiera_base_plus.pt
└── data/
    └── foodmix_sa1b/                 # SA-1B形式に変換済みのデータ
        ├── images/                    # 7,118枚の画像
        ├── annotations/               # RLE形式のマスクJSON
        ├── train.txt                  # 4,983枚
        └── val.txt                    # 2,135枚
```

### train.pyの該当部分（line 245）
```python
if __name__ == "__main__":
    initialize_config_module("sam2", version_base="1.2")  # ここでsam2パッケージを検索パスに設定
    parser = ArgumentParser()
    # ...
    args = parser.parse_args()
    register_omegaconf_resolvers()
    main(args)
```

### main関数内（line 124）
```python
def main(args) -> None:
    cfg = compose(config_name=args.config)  # ここでエラー発生
```

### 試した実行コマンド（すべて失敗）
```bash
# 1. フルパス指定
python training/train.py \
  -c sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml \
  --use-cluster 0 --num-gpus 1

# 2. 相対パス指定
python training/train.py \
  -c sam2.1_training/sam2.1_hiera_b+_foodmix_finetune \
  --use-cluster 0 --num-gpus 1

# 3. 公式の設定ファイルも同様に失敗
python training/train.py \
  -c sam2.1_training/sam2.1_hiera_b+_MOSE_finetune \
  --use-cluster 0 --num-gpus 1
```

### SAM2のインストール状況
```bash
# 最初のインストール
cd external/sam2
pip install -e ".[dev]"  # 実行済み

# 再インストール試行（タイムアウト）
pip install -e . --break-system-packages
```

### カスタム設定ファイルの主要部分
```yaml
# @package _global_

scratch:
  resolution: 1024
  train_batch_size: 1
  num_frames: 1
  max_num_objects: 50
  base_lr: 1.0e-4
  vision_lr: 5.0e-5
  num_epochs: 40

dataset:
  img_folder: ../../data/foodmix_sa1b/images
  gt_folder: ../../data/foodmix_sa1b/annotations
  file_list_txt: ../../data/foodmix_sa1b/train.txt
```

## 質問

1. Hydraの`initialize_config_module("sam2", version_base="1.2")`で、pkg://sam2の検索パスにカスタム設定ファイルを含める正しい方法は何ですか？

2. SAM2.1の公式実装で、external/sam2ディレクトリから`pip install -e .`でインストールした後、sam2/configs/sam2.1_training/内の新しい設定ファイルが認識されない問題を解決する方法は？

3. Hydraの設定検索パスに、ローカルのconfigディレクトリを追加する方法、または`@package _global_`ディレクティブを正しく使用してHydraに設定ファイルを認識させる方法は？

4. 代替案として、Hydraを使わずにSAM2.1の学習を実行する方法（training/train.pyを改変せずに）はありますか？

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。

