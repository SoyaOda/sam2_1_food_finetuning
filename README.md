# SAM2.1 Food Segmentation Fine-tuning

SAM 2.1を使用して、FoodSeg103（材料レベル）とUEC-FoodPix Complete（料理レベル）データセットでインスタンス分割モデルを学習するプロジェクトです。

## 🎯 プロジェクト概要

- **目的**: 食品画像の高精度なインスタンスセグメンテーション
- **モデル**: Meta SAM 2.1 (Segment Anything Model v2.1)
- **データセット**:
  - FoodSeg103: 材料レベルのピクセル単位ラベル
  - UEC-FoodPix Complete: 料理レベルのセグメンテーション

## 📋 要件

- Python 3.10以上
- PyTorch 2.5.1以上
- CUDA対応GPU（推奨: 16GB以上のVRAM）
- 50GB以上のディスク空き容量

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 環境構築とSAM2.1のインストール
bash scripts/00_setup_env.sh
```

### 2. データセット準備

```bash
# FoodSeg103のダウンロード（研究用途のみ）
bash scripts/01_download_foodseg103.sh

# UEC-FoodPix Completeのダウンロード（非商用研究用途のみ）
bash scripts/02_download_uecfoodpix.sh
```

### 3. データ前処理

```bash
# FoodSeg103をSA-1B形式に変換
python scripts/10_prepare_foodseg103.py

# UEC-FoodPixをSA-1B形式に変換
python scripts/11_prepare_uecfoodpix.py

# データセット統合とtrain/valスプリット作成
python scripts/12_merge_to_sa1b.py

# (オプション) 画像を1024x1024にリサイズ
python scripts/13_optional_resize_1024.py
```

### 4. 学習実行

```bash
# メモリ最適化版SAM2.1の微調整を開始（推奨）
bash scripts/train_with_monitoring.sh --memory-optimized --auto-resume

# 通常版（従来）
bash scripts/20_train_food_sam2.sh
```

### 5. 評価と可視化

```bash
# 学習済みモデルの評価
python scripts/21_eval_and_viz.py --visualize
```

## 📁 プロジェクト構造

```
sam2_1_food_finetuning/
├── README.md
├── CLAUDE.md                      # Claude Code用の指示書
├── external/
│   └── sam2/                     # SAM2.1公式リポジトリ
├── data/
│   ├── FoodSeg103/               # FoodSeg103データセット
│   ├── UECFOODPIXCOMPLETE/       # UEC-FoodPixデータセット
│   └── foodmix_sa1b/             # SA-1B形式に変換済みデータ
│       ├── images/               # 画像ファイル
│       ├── annotations/          # JSONアノテーション（RLE形式）
│       ├── train.txt             # 訓練用ファイルリスト
│       └── val.txt               # 検証用ファイルリスト
├── scripts/
│   ├── 00_setup_env.sh                   # 環境セットアップ
│   ├── 01_download_foodseg103.sh         # FoodSeg103ダウンロード
│   ├── 02_download_uecfoodpix.sh         # UEC-FoodPixダウンロード
│   ├── 10_prepare_foodseg103.py          # FoodSeg103前処理
│   ├── 11_prepare_uecfoodpix.py          # UEC-FoodPix前処理
│   ├── 12_merge_to_sa1b.py               # データ統合
│   ├── 13_optional_resize_1024.py        # 画像リサイズ
│   ├── 20_train_food_sam2.sh             # 学習実行（従来版）
│   ├── 21_eval_and_viz.py                # 評価・可視化
│   ├── train_sam2_memory_optimized.py    # メモリ最適化学習スクリプト
│   ├── train_with_monitoring.sh          # システム監視付き学習
│   ├── resume_training.py                # 自動再開機能
│   └── create_training_config.sh         # 学習設定生成
└── configs/
    └── sam2.1_training/
        └── sam2.1_hiera_b+_foodmix_finetune.yaml  # 学習設定
```

## ⚙️ 詳細設定

### 学習パラメータの調整

`configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml`を編集：

- **バッチサイズ**: GPUメモリに応じて調整（デフォルト: 1）
- **学習率**: 1e-4（必要に応じて1e-5〜3e-4で調整）
- **エポック数**: 40（デフォルト）
- **精度**: bf16（A100等）/ fp32（T4等）

### GPUメモリ別推奨設定

| GPU | VRAM | バッチサイズ | 精度 | メモリ最適化 |
|-----|------|------------|------|------------|
| T4 | 16GB | 1 | fp32 | 必須 |
| V100 | 32GB | 2 | fp32 | 推奨 |
| A100 | 40GB | 4 | bf16 | オプション |

### メモリ最適化機能

- **メモリクリーンアップ**: 25ステップごとに自動実行
- **ガベージコレクション**: 10ステップごとに自動実行
- **GPU監視**: 使用率85%を超えるとアラート
- **自動再開**: システムクラッシュ後の自動復旧
- **チェックポイント**: 500ステップごとに保存

## 📊 データセット情報

### FoodSeg103
- **クラス数**: 103（材料レベル）
- **画像数**: 約7,000枚
- **ライセンス**: 研究用途
- **パスワード**: LARCdataset9947

### UEC-FoodPix Complete
- **クラス数**: 102（料理レベル）
- **画像数**: 約10,000枚
- **ライセンス**: 非商用研究用途のみ
- **特徴**: Rチャネルにクラス情報

## 🔍 トラブルシューティング

### よくあるエラーと対処法

1. **GPUメモリ不足**
   ```bash
   # バッチサイズを1に設定
   # configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yamlを編集
   batch_sizes: [1]
   ```

2. **データセットダウンロード失敗**
   ```bash
   # 手動でダウンロードして配置
   # FoodSeg103: https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip
   # UEC-FoodPix: https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar
   ```

3. **SAM2インポートエラー**
   ```bash
   cd external/sam2
   pip install -e ".[dev]"
   ```

## 📝 注意事項

- **ライセンス**: データセットは研究用途限定です。商用利用はできません。
- **引用**: 学術論文等で使用する場合は、各データセットの引用が必要です。
- **計算リソース**: 完全な学習には相当な計算時間が必要です（GPU環境で数時間〜数日）。

## 🔧 カスタマイズ

### 独自データセットの追加

1. 画像とマスクを準備
2. `scripts/10_prepare_foodseg103.py`を参考に前処理スクリプトを作成
3. SA-1B形式（RLEエンコード）に変換
4. `scripts/12_merge_to_sa1b.py`でデータ統合

### モデルアーキテクチャの変更

- `sam2.1_hiera_tiny.pt`: 軽量版
- `sam2.1_hiera_base_plus.pt`: バランス版（デフォルト）
- `sam2.1_hiera_large.pt`: 高精度版

## 📚 参考資料

- [SAM 2.1 公式リポジトリ](https://github.com/facebookresearch/sam2)
- [FoodSeg103 論文](https://xiongweiwu.github.io/foodseg103.html)
- [UEC-FoodPix Complete](https://mm.cs.uec.ac.jp/uecfoodpix/)

## 🤝 貢献

プルリクエストや問題報告は歓迎します。大きな変更を行う場合は、まずissueを開いて議論してください。

## 📄 ライセンス

このプロジェクトのコードはMITライセンスです。ただし、使用するデータセットとSAM2.1モデルには各自のライセンスが適用されます。

---

**注**: このプロジェクトは研究目的で作成されています。商用利用の際は各コンポーネントのライセンスを確認してください。