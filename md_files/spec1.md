以下は、\*\*「SAM 2.1 を中核に、FoodSeg103（材料レベル）＋UEC‑FoodPix Complete（料理レベル）でインスタンス分割を学習」\*\*するための、**曖昧性ゼロ**の実装計画（ダウンロード〜前処理〜学習〜検証まで、フルコード付き）です。Colab（推奨）／ローカルGPUのどちらでも実行できます。

> **要点**
>
> * **SAM 2.1**は公式リポジトリで**学習／微調整コード**が公開済み（2024‑09‑30の開発者向けスイートに含まれる）。最低要件は Python ≥3.10、PyTorch ≥2.5.1。チェックポイントの自動DLスクリプトと**学習用Hydra設定**が同梱されています。([GitHub][1])
> * **FoodSeg103**は研究室の配布ページ／GitHubに**ZIPと解凍パスワード**記載あり（`LARCdataset9947`）。材料の**ピクセル単位ラベル**付き。([GitHub][2])
> * **UEC‑FoodPix Complete**は公式ページから直接ダウンロード可。**RチャネルにクラスIDが入ったPNGマスク**、`train.txt`／`test.txt`や `category.txt` を含む構造が明示されています（**非商用研究目的**に限定）。([Yanai Lab][3])
> * SAM2 の学習コードは**画像（SA‑1B 形式）**／**動画（DAVIS系）**を両対応。画像学習では**SA‑1B風（画像ごとJSON＋RLE）**を例として実装されているため、FoodSeg103/UECの**セマンティックマスク→連結成分ごとのオブジェクトマスク**に分解してSA‑1B形式へ正規化し、そのまま学習に使用します。([Hugging Face][4], [GitHub][5])

---

## 0) 成果物（プロジェクト構成）

```text
foodsammix/
├─ README.md
├─ external/
│   └─ sam2/                       # Meta公式 SAM 2.1 (サブモジュール or クローン)
├─ data/
│   ├─ FoodSeg103/                 # 食材ラベル付きデータ（ZIP解凍先）
│   ├─ UECFOODPIXCOMPLETE/         # 料理ラベル付きデータ（TAR解凍先）
│   └─ foodmix_sa1b/               # SA-1B風に正規化した学習用データ
│       ├─ images/                 # 画像（必要なら 1024x1024 リサイズ済み）
│       ├─ annotations/            # 画像ごと JSON（RLEマスクの配列）
│       ├─ train.txt               # 学習用 画像ファイル名（拡張子込み）
│       └─ val.txt                 # 検証用 画像ファイル名
├─ scripts/
│   ├─ 00_setup_env.sh
│   ├─ 01_download_foodseg103.sh
│   ├─ 02_download_uecfoodpix.sh
│   ├─ 10_prepare_foodseg103.py
│   ├─ 11_prepare_uecfoodpix.py
│   ├─ 12_merge_to_sa1b.py
│   ├─ 13_optional_resize_1024.py
│   ├─ 20_train_food_sam2.sh
│   ├─ 21_eval_and_viz.py
│   └─ 22_export_inference_colab.py
└─ configs/
    └─ sam2.1_training/
        └─ sam2.1_hiera_b+_foodmix_finetune.yaml   # MOSEサンプルを複製して編集
```

---

## 1) 環境構築（Colab / ローカルGPU）

### 1‑A. 依存関係とSAM2.1の導入

```bash
# 00_setup_env.sh
set -eux

# (Colabなら不要) 新規venv or conda推奨
# conda create -n foodsammix python=3.10 -y && conda activate foodsammix

# pytorch 2.5.1+ を環境に合わせて (以下は例。CUDA版は https://pytorch.org/ を参照)
pip install --upgrade pip
# GPU環境に合わせてインストールしてください（例: cu124）・・・ほかのプロジェクトでSAM2.1使えているので現状の環境で問題なさそう
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 補助ライブラリ
pip install opencv-python numpy pycocotools scikit-image pillow tqdm hydra-core omegaconf matplotlib

# SAM2.1 本体
mkdir -p external && cd external
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[dev]"   # 開発/学習向け依存を含めてインストール
# チェックポイント (sam2.1 各種)
cd checkpoints && ./download_ckpts.sh && cd ..
```

* SAM 2.1 の**学習コード**・**チェックポイント**・\*\*要件（Python3.10 / Torch2.5.1+）\*\*は公式READMEに明記されています。([GitHub][1])

---

## 2) データセットの取得

> **ライセンス注意**
>
> * **UEC‑FoodPix/Complete**は**非商用研究**のみ利用可。研究用途であることを確認してください。([Yanai Lab][3])
> * **FoodSeg103**は研究目的配布。公式リポの指示に従い、ZIPと**パスワード**を用いて取得します。([GitHub][2])

### 2‑A. FoodSeg103

```bash
# 01_download_foodseg103.sh
set -eux
cd "$(git rev-parse --show-toplevel)"

mkdir -p data/FoodSeg103 && cd data/FoodSeg103
# 公式配布ZIP（GitHub READMEにURLとパスワードが記載）
# URL例: https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip
# パスワード: LARCdataset9947
curl -L -o FoodSeg103.zip "https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip"
unzip -P LARCdataset9947 FoodSeg103.zip
```

* **GitHubのREADME**に「`FoodSeg103.zip` を `./data/FoodSeg103/` に展開し、パスワード`LARCdataset9947`」と明記。([GitHub][2])
* データは**材料クラスのピクセルラベル**を含みます（Dataset概要）。([Xiongwei's Homepage][6])

### 2‑B. UEC‑FoodPix Complete

```bash
# 02_download_uecfoodpix.sh
set -eux
cd "$(git rev-parse --show-toplevel)"

mkdir -p data && cd data
# 公式DLリンク (サイトに直リンクが記載)
# https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar
curl -L -o UECFOODPIXCOMPLETE.tar "https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar"
tar -xf UECFOODPIXCOMPLETE.tar
```

* サイト上に**Downloads**の直リンク（`.tar`）と**ディレクトリ構造**（`img/`, `mask/`, `train.txt`, `test.txt`, `category.txt`）が明示。マスクは**RチャネルにクラスID**。([Yanai Lab][3])

---

## 3) SA‑1B 風（画像ごとJSON＋RLE）への正規化

SAM2.1 の画像学習は、**SA‑1B 風のアノテーション（画像ごとJSON／RLEマスク配列）**のサンプル実装が用意されています。ここでは FoodSeg103/UEC の**セマンティックマスク**を**連結成分ごとに分解**して、\*\*クラス非依存の“オブジェクトマスク群”\*\*に変換し、**SA‑1B と同じRLE構造**で保存します（`annotations/<image_basename>.json`）。([Hugging Face][4], [GitHub][5])

> **ポイント**：FoodSeg103/UECは「ピクセルごとにクラスID」です。**各クラス内の連結成分を1インスタンス**と見なし、背景（ID=0）は除外します。UECは**Rチャネルの画素値**がIDなのでそれを読みます。([Yanai Lab][3])

```python
# scripts/10_prepare_foodseg103.py
import os, json, glob
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from pycocotools import mask as mask_utils
from tqdm import tqdm

def rle_of_binary_mask(bin_mask: np.ndarray):
    # Fortran order RLE
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def extract_instances_from_semantic(mask_arr: np.ndarray, ignore_ids={0}):
    h, w = mask_arr.shape
    ann_list = []
    ann_id = 0
    valid_ids = [i for i in np.unique(mask_arr) if i not in ignore_ids]
    for cid in valid_ids:
        comp = (mask_arr == cid).astype(np.uint8)
        lab = label(comp, connectivity=1, background=0)
        for lid in np.unique(lab):
            if lid == 0: continue
            inst = (lab == lid)
            area = int(inst.sum())
            if area < 10:  # 極小除外
                continue
            ys, xs = np.where(inst)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            bbox = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
            rle = rle_of_binary_mask(inst)
            ann_list.append({
                "id": ann_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
                # SA-1Bと互換の補助項目（適当値でも可）
                "predicted_iou": 1.0,
                "stability_score": 1.0,
                "crop_box": [0, 0, w, h],
                "point_coords": [[int((x0+x1)/2), int((y0+y1)/2)]],
            })
            ann_id += 1
    return ann_list

def main():
    root = "data/FoodSeg103"   # ZIP展開先
    # 画像・マスクの対（パス）は実際の展開構造に合わせて列挙
    # 例: images/*.jpg と masks/*.png のような一般的配置を総当たり探索
    img_paths = []
    mask_paths = {}
    for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        base = os.path.basename(p)
        if base.lower().endswith((".jpg", ".jpeg", ".png")):
            if "mask" in p.lower() or "ann" in p.lower():
                mask_paths[base.rsplit(".",1)[0]] = p
            else:
                img_paths.append(p)

    out_img_dir = "data/foodmix_sa1b/images"
    out_ann_dir = "data/foodmix_sa1b/annotations"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    for ip in tqdm(img_paths):
        base = os.path.basename(ip)
        stem = base.rsplit(".",1)[0]
        if stem not in mask_paths:
            continue
        mp = mask_paths[stem]
        # 読み込み
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp)
        mask_arr = np.array(mask)
        # FoodSeg103は一般に整数IDの単一チャンネルPNG想定
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:,:,0]
        h, w = mask_arr.shape
        anns = extract_instances_from_semantic(mask_arr, ignore_ids={0})
        # SA-1B風JSON
        out = {
            "image": {"image_id": 0, "width": int(w), "height": int(h), "file_name": base},
            "annotations": anns
        }
        # コピー保存
        img.save(os.path.join(out_img_dir, base))
        with open(os.path.join(out_ann_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)

if __name__ == "__main__":
    main()
```

```python
# scripts/11_prepare_uecfoodpix.py
import os, json, glob
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from skimage.measure import label
from tqdm import tqdm

def rle_of_binary_mask(bin_mask: np.ndarray):
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def main():
    root = "data/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLTE"
    img_dir = os.path.join(root, "train", "img")
    msk_dir = os.path.join(root, "train", "mask")
    out_img_dir = "data/foodmix_sa1b/images"
    out_ann_dir = "data/foodmix_sa1b/annotations"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    for ip in tqdm(img_paths):
        base = os.path.basename(ip)        # 例: 1234.jpg
        stem = base[:-4]
        mp = os.path.join(msk_dir, f"{stem}.png")
        if not os.path.exists(mp): 
            continue
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp)
        m = np.array(mask)
        # 公式ページの仕様: RチャネルにクラスID（背景0）:contentReference[oaicite:12]{index=12}
        if m.ndim == 3:
            m = m[:,:,0]

        H, W = m.shape
        anns = []
        ann_id = 0
        for cid in np.unique(m):
            if cid == 0:  # 背景スキップ
                continue
            lab = label((m==cid).astype(np.uint8), connectivity=1, background=0)
            for lid in np.unique(lab):
                if lid == 0: continue
                inst = (lab==lid)
                area = int(inst.sum())
                if area < 10:
                    continue
                ys, xs = np.where(inst)
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                bbox = [int(x0), int(y0), int(x1-x0+1), int(y1-y0+1)]
                anns.append({
                    "id": ann_id,
                    "segmentation": rle_of_binary_mask(inst),
                    "bbox": bbox,
                    "area": area,
                    "predicted_iou": 1.0,
                    "stability_score": 1.0,
                    "crop_box": [0,0,W,H],
                    "point_coords": [[int((x0+x1)/2), int((y0+y1)/2)]],
                })
                ann_id += 1

        out = {"image": {"image_id": 0, "width": int(W), "height": int(H), "file_name": base},
               "annotations": anns}
        img.save(os.path.join(out_img_dir, base))
        with open(os.path.join(out_ann_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)

if __name__ == "__main__":
    main()
```

### 3‑A. 学習・検証スプリットの作成

```python
# scripts/12_merge_to_sa1b.py
import os, random, glob
random.seed(42)

root = "data/foodmix_sa1b"
imgs = sorted(glob.glob(os.path.join(root, "images", "*.jpg"))) + \
       sorted(glob.glob(os.path.join(root, "images", "*.png")))

# 画像ごとに annotations/<stem>.json が存在することを確認
pairs = []
for ip in imgs:
    stem = os.path.splitext(os.path.basename(ip))[0]
    jp = os.path.join(root, "annotations", f"{stem}.json")
    if os.path.exists(jp):
        pairs.append(os.path.basename(ip))

random.shuffle(pairs)
n = len(pairs)
train = pairs[: int(n*0.9)]
val   = pairs[int(n*0.9):]

with open(os.path.join(root, "train.txt"), "w") as f:
    f.write("\n".join(train))
with open(os.path.join(root, "val.txt"), "w") as f:
    f.write("\n".join(val))

print(f"Train: {len(train)}, Val: {len(val)}")
```

---

## 4) 画像サイズの正規化（任意／推奨）

Roboflowの実地ガイドでは、**学習設定に合わせて 1024×1024 にストレッチ**したデータで微調整する例が示されています。SAM2.1の学習設定によっては**入力解像度を固定**する前提があるため、安定運用のために事前リサイズを選ぶ場合は以下を実行してください。([Roboflow Blog][7])

```python
# scripts/13_optional_resize_1024.py
import os, glob
from PIL import Image
from tqdm import tqdm

IN_DIR = "data/foodmix_sa1b/images"
OUT_DIR = IN_DIR  # 上書き (別ディレクトリを使いたい場合は変更)

for p in tqdm(glob.glob(os.path.join(IN_DIR, "*.*"))):
    im = Image.open(p).convert("RGB").resize((1024,1024), Image.BILINEAR)
    im.save(p)
```

> 公式SAM2学習コードは\*\*Hydra設定で変換（transforms）\*\*を持ち、SA‑1Bや動画データの例が付属しています（`training/dataset/`と設定YAML）。本手順では、**事前リサイズ**で設定側の依存を減らしています。([Hugging Face][4])

---

## 5) SAM 2.1 の学習設定（Hydra YAML）

SAM2.1 リポの**MOSE微調整の設定YAML**を**複製してパスだけ差し替える**のが最も確実です（`configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml` を `..._foodmix_finetune.yaml` にコピー）。SAM2.1 公式の **`training/train.py`** は**画像データ**に対して **`SA1BRawDataset`** を例示しています。以下は**最小限の置換**例です：([Hugging Face][4])

```yaml
# configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml
# (MOSEサンプルをコピーして下記キーを編集)

experiment_log_dir: ./sam2_logs/foodmix_bplus_40ep

model:
  checkpoint: ./checkpoints/sam2.1_hiera_base_plus.pt
  model_cfg: configs/sam2.1/sam2.1_hiera_b+.yaml
  # 追加でfreeze等のパラメータがある場合は既存設定に従う

data:
  train:
    _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset
    phases_per_epoch: 1
    batch_sizes: [1]     # 単GPU/T4相当なら 1 を推奨
    datasets:
      - _target_: training.dataset.vos_dataset.VOSDataset
        training: true
        video_dataset:
          _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
          img_folder: data/foodmix_sa1b/images
          gt_folder:  data/foodmix_sa1b/annotations
          file_list_txt: data/foodmix_sa1b/train.txt
        sampler:
          _target_: training.dataset.vos_sampler.RandomUniformSampler
          num_frames: 1
          max_num_objects: 50
        transforms: ${image_transforms}   # 既定の画像変換を流用

  val:
    _target_: training.dataset.vos_dataset.VOSDataset
    training: false
    video_dataset:
      _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
      img_folder: data/foodmix_sa1b/images
      gt_folder:  data/foodmix_sa1b/annotations
      file_list_txt: data/foodmix_sa1b/val.txt
    sampler:
      _target_: training.dataset.vos_sampler.CenterFrameSampler
      num_frames: 1
      max_num_objects: 50
    transforms: ${image_transforms}

trainer:
  max_epochs: 40
  num_workers: 4
  precision: "bf16"       # A100等。T4なら "fp32" に変更
  grad_accum_steps: 1

optimizer:
  name: AdamW
  lr: 1.0e-4              # まずは標準値。必要に応じて 1e-5〜3e-4 で探索
  weight_decay: 0.05
  betas: [0.9, 0.999]
  # パラメーターグループ（エンコーダはLR小さめ等）が既定で定義されている場合はそれに従う
```

* **学習の起動コマンド**（公式READMEの手順通り）：

  ```bash
  cd external/sam2
  python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml \
    --use-cluster 0 --num-gpus 1
  ```

  （A100等マルチGPUなら `--num-gpus 8`／SLURMオプションも可）([Hugging Face][4])

> 参考：RoboflowのSAM‑2.1微調整ガイドも、**40エポック**・**1024入力**・**`training/train.py` 実行**などの実例を示しています。([Roboflow Blog][7])

---

## 6) 学習確認・可視化・簡易評価

```python
# scripts/21_eval_and_viz.py
import os, json, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ログディレクトリに保存された微調整済みチェックポイントを指定
EXP_DIR = "external/sam2/sam2_logs/foodmix_bplus_40ep"
CKPT = os.path.join(EXP_DIR, "checkpoints", "checkpoint.pt")
CFG  = "configs/sam2.1/sam2.1_hiera_b+.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2 = build_sam2(CFG, CKPT, device=device)
predictor = SAM2ImagePredictor(sam2)

# バリデーションからランダムに可視化
val_txt = "data/foodmix_sa1b/val.txt"
with open(val_txt) as f:
    names = [ln.strip() for ln in f if ln.strip()]
name = random.choice(names)
img_path = os.path.join("data/foodmix_sa1b/images", name)
ann_path = os.path.join("data/foodmix_sa1b/annotations", name.rsplit(".",1)[0]+".json")

img = np.array(Image.open(img_path).convert("RGB"))
predictor.set_image(img)
# 自動マスク生成でも良いが、ここではポイントなし（全体サーチ）
masks, _, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)

# GTとの簡易IoU（最大IoUをとる粗い計測）
with open(ann_path) as f:
    gt = json.load(f)["annotations"]

def rle_to_bin(rle, h, w):
    from pycocotools import mask as mask_utils
    rle = rle.copy()
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    m = mask_utils.decode(rle)
    return m.reshape((h,w), order="F")

H,W = img.shape[:2]
gts = [rle_to_bin(a["segmentation"], H, W).astype(bool) for a in gt]

def iou(a,b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return (inter/union) if union>0 else 0.0

# 可視化
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title("Image"); plt.imshow(img); plt.axis('off')
plt.subplot(1,2,2); plt.title("Pred masks"); 
overlay = img.copy()
for m in masks:
    binm = m.astype(bool)
    overlay[binm] = (overlay[binm]*0.3 + np.array([255,0,0])*0.7).astype(np.uint8)
plt.imshow(overlay); plt.axis('off')
plt.show()

# 粗いIoU評価
pred_bins = [m.astype(bool) for m in masks]
best_ious = []
for gb in gts:
    best_ious.append(max([iou(gb, pb) for pb in pred_bins] + [0.0]))
print(f"Mean best IoU (rough): {np.mean(best_ious):.3f}")
```

---

## 7) 実運用パラメータ（推奨初期値）

* **バックボーン**：`sam2.1_hiera_base_plus.pt`（精度と速度のバランス）。`sam2.1_hiera_large.pt` は精度重視。([GitHub][1])
* **入力解像度**：まずは 1024×1024 固定（前処理リサイズ）。設定で可変にする場合は `transforms` を既定から流用。([Roboflow Blog][7])
* **バッチサイズ**：単GPU/T4なら 1、A100なら 2–4。
* **エポック**：40（初期探索の目安）。([Roboflow Blog][7])
* **最適化**：AdamW、`lr=1e-4`（必要に応じ `1e-5〜3e-4` で格子探索）。
* **精度モード**：A100等は `bf16`、T4系は `fp32`（設定YAMLの `precision`）。([Hugging Face][4])
* **最大物体数**：`max_num_objects=50`（盛付けの細粒度を考慮し高めに）

---

## 8) よくある落とし穴と対策

1. **学習用JSONのRLE**
   SA‑1B形式の**RLEはFortran order**でエンコード（`pycocotools.encode(np.asfortranarray(...))`）すること。([GitHub][5])
2. **UECのマスク解釈**
   **Rチャネルのみ**がクラスID。RGB全体で扱わない。([Yanai Lab][3])
3. **ライセンス**
   UECは**非商用研究**限定。成果公開時は**引用**も忘れずに（サイトにBibTeX例あり）。([Yanai Lab][3])
4. **学習コードの場所**
   SAM2.1 リポの `training/` に**データセットクラス**・**Trainer**・**optimizer** 等があり、`training/train.py` で起動。Hydra設定の**パスだけ正しく**。([Hugging Face][4])
5. **入出力回り**
   画像名と `annotations/<stem>.json` の**整合**、`train.txt/val.txt` の**改行**・**拡張子**に注意。

---

## 9) 追加メモ（拡張）

* **カテゴリ情報を保持したい場合**：JSONの各annotationに独自キー（例 `{"category": <id>}`）を足しておいても、SAM2学習のマスク品質には影響しません（クラス非依存学習）。
* **評価の厳密化**：COCO‑style AP（IoUしきい値系列）やBoundary F‑scoreの計測ツールを追加実装可。
* **学習スキーマ**：材料レベル（FoodSeg103）と料理レベル（UEC）を**混在**させることで、「食材／具材の細かい境界」と「料理輪郭」の両方に強いモデルを狙えます。

---

## 10) 実行手順まとめ

```bash
# 0. 環境
bash scripts/00_setup_env.sh

# 1. データ取得
bash scripts/01_download_foodseg103.sh
bash scripts/02_download_uecfoodpix.sh

# 2. SA‑1B形式に正規化
python scripts/10_prepare_foodseg103.py
python scripts/11_prepare_uecfoodpix.py
python scripts/12_merge_to_sa1b.py

# 3. （任意）1024x1024 へ事前リサイズ
python scripts/13_optional_resize_1024.py

# 4. 学習（設定YAMLを用意済みとする）
bash scripts/20_train_food_sam2.sh
# (または)
cd external/sam2
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml \
  --use-cluster 0 --num-gpus 1

# 5. 検証・可視化
python scripts/21_eval_and_viz.py
```

---

### 参考・根拠

* **SAM 2.1 公式**：学習コード・チェックポイント・要件、SAM 2.1 の公開（2024‑09‑30、Training/README 参照）、使用例・コンフィグパス。([GitHub][1])
* **SA‑1B 形式**：画像ごとJSON＋RLEの構造（`image`／`annotations`）、RLEはCOCO形式。([GitHub][5])
* **FoodSeg103**：公式HPとGitHub。ZIP配布とパスワード、材料ピクセルラベル。([Xiongwei's Homepage][6], [GitHub][2])
* **UEC‑FoodPix Complete**：ダウンロードとファイル構造、**Rチャネルがラベル**、利用は**非商用研究目的**。([Yanai Lab][3])
* **実地ガイド**：SAM‑2.1 微調整手順（1024入力、40エポック、`training/train.py`で学習）。([Roboflow Blog][7])

---