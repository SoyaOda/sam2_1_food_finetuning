# SAM2.1学習設定に関するQuery

## 調査が必要な内容

SAM2.1の学習設定YAMLファイルの正確な構造について、以下の点を調査する必要があります：

### 1. SAM2.1公式の学習設定構造
- `training/train.py`で使用されるHydra設定の正確なスキーマ
- SA1BRawDatasetクラスの必要なパラメータ
- image_transformsの定義方法

### 2. 微調整時の推奨設定
- MOSEファインチューニング設定の具体的な内容
- バックボーンのフリーズ設定
- 学習率スケジューラの設定

### 3. データローダー設定
- TorchTrainMixedDatasetの設定方法
- VOSDatasetとSA1BRawDatasetの組み合わせ方
- バッチサイズとGPUメモリの関係

## Query Prompt

以下について、SAM2.1の公式実装（https://github.com/facebookresearch/sam2）を参考にした正確な情報を教えてください：

1. SAM2.1のtraining/train.pyで使用される学習設定YAMLファイルの正確な構造（特にMOSE微調整の例）
2. SA1BRawDatasetクラスを使用する際の必要なパラメータとディレクトリ構造
3. 画像データ（静止画）の学習時のimage_transformsの定義
4. 微調整時のモデルパラメータのフリーズ設定とoptimizer設定
5. Hydra設定ファイルでのデータパスの相対パス/絶対パスの指定方法

これらについて、SAM2.1等の公式実装を参考にした解決策を教えてください。