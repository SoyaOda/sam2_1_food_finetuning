# 推奨コマンド一覧

## 環境セットアップ
```bash
# Python環境作成
conda create -n foodsammix python=3.10 -y
conda activate foodsammix

# PyTorch インストール（CUDA版）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

## データセット取得
```bash
# FoodSeg103ダウンロード
curl -L -o FoodSeg103.zip "https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip"
unzip -P LARCdataset9947 FoodSeg103.zip

# UEC-FoodPix Complete ダウンロード
curl -L -o UECFOODPIXCOMPLETE.tar "https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar"
tar -xf UECFOODPIXCOMPLETE.tar
```

## 学習実行
```bash
cd external/sam2
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml \
  --use-cluster 0 --num-gpus 1
```

## データ前処理
```bash
python scripts/10_prepare_foodseg103.py
python scripts/11_prepare_uecfoodpix.py
python scripts/12_merge_to_sa1b.py
python scripts/13_optional_resize_1024.py
```

## 評価・可視化
```bash
python scripts/21_eval_and_viz.py
```

## Git操作
```bash
git status
git add .
git commit -m "message"
git push origin main
```

## システムコマンド（Linux）
- ls: ファイル一覧
- cd: ディレクトリ移動
- mkdir -p: ディレクトリ作成
- cp: ファイルコピー
- rm: ファイル削除