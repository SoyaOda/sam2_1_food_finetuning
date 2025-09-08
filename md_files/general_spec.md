様々なユーザーの投稿する料理画像とその画像をSAM2.1とUniDepth v2で変換した情報をもちいて、VLM（Qwen2.5 VLなど）に入力して料理画像に含まれるすべての料理と食材・材料の量とカロリーをできる限り正確に推定したい　どのようにするのがベスト？
ChatGPT:
Request for GPT-5 Pro
Reasoned for 8m 48s

最高精度を狙うなら、幾何（体積）×意味（材料・料理）×栄養DBを“拘束”し合う三層融合パイプラインがベストです。具体的には SAM 2.1 で確実にインスタンス分割し、UniDepth v2 でメートル尺度の深度とカメラ内参を推定、Qwen2.5‑VL で料理名・材料・配合を推論し、最後に密度・調理収縮/保持率と栄養成分表でグラム量とカロリーに落とし込む、という流れです（モデル能力の要点：SAM 2.1 の改良チェックポイント“2.1”と学習/推論実装、Qwen2.5‑VLのボックス/ポイント出力と安定JSON、UniDepth v2の単眼メトリック深度＋信頼度＋内参推定などが鍵）。
GitHub
+1
Amazon Web Services, Inc.
Qwen
Hugging Face

全体アーキテクチャ（2025年版・実運用向け）

入力（複数画像）

Qwen2.5‑VL は複数画像をインターリーブして扱えるので、以下を同時投入：
A: 元画像、B: セグメントIDを描いたマスク重畳画像、C: カラー深度ヒートマップ、D: 高さマップ（皿面からの相対高さ）。指示文にはID→料理/食材をJSONで返すこと、各IDの**性状（固形/液体/葉物/粒状）と主材料比率（%）**も返すことを明記。
Hugging Face

候補検出→分割

まず Qwen2.5‑VL に簡易検出（BBox/ポイント）を出させ、それをSAM 2.1のプロンプトとして渡して各料理・具材をインスタンス分割。食品以外（皿、箸、影、装飾）は除外タグに。SAM 2.1 は2.1系の改良チェックポイントと学習コードが公開されており、食領域に合わせて軽微なドメインFT（数千枚規模）を行うと安定します。
Qwen
GitHub

深度→体積（幾何トラック）

UniDepth v2でメトリック深度と内参を取得。皿/テーブル平面をRANSACでフィットし、各マスクの高さマップ h(x,y) を得て、体積 V ≈ Σ h(x,y)·a_pix を計算。UniDepth v2 は信頼度マップも出せるので、体積積分を信頼度重み付きで行い外れ値を抑えます。
GitHub

意味/材料・配合（セマンティックトラック）

各マスクの切り出し画像＋全体画像＋（任意で）近縁レシピのテキスト（後述）を Qwen2.5‑VL に供給。

出力は料理名→構成材料（上位k、%）→調理法。Qwenの位置合わせJSONを併用するとIDと出力の整合が取りやすいです。
Qwen

量（g）とカロリー（kcal）への写像

密度表（FAO/INFOODS Density DB 等）から材料タイプ別密度 ρを選択（固形・液体・粒状・葉物で分岐、料理固有は優先ルールで上書き）。質量 m = ρ·V。
FAOHome

調理による重量変化率/保持率で補正（例：茹で・炒めでの水分/栄養変化）。日本は重量変化率の定義・運用が明示されているのでそれに準拠。
Food Database

栄養DBは日本向けは日本食品標準成分表2020（八訂）（必要に応じ増補2023）を第一候補、外食/輸入品やブランド品はUSDA FoodData Centralをフォールバック。
MEXT
Food Database
FoodData Central

統合推定（拘束最適化）

目的関数例：
L = 体積由来質量 − レシピ由来質量 の整合 + 典型盛付け量との偏差 + VLMの信頼度に基づく正則化。

レシピ拘束には Recipe1M+ 等の画像↔レシピ埋め込みで近傍レシピを引いて材料比率の事前分布を与えるのが効きます。
Recipe1M
arXiv

不確実性（CI）

UniDepthの信頼度、密度表のばらつき、VLMの材料比率分散から分解寄与を出し、各アイテムの量[g]とkcalの95%CIを付与。

実装ディテール（要点）
A. 分割とデータセット

インスタンス分割は SAM 2.1 を中核に。FoodSeg103（Ingredients/Pixel）とUEC‑FoodPix Complete（料理ピクセル）で微調整 or 合成学習が有効。前者は材料レベル、後者は料理レベルのマスクがあり、組み合わせで階層分割の足場になります。
Xiongwei's Homepage
+1
Yanai Lab
ACM Digital Library

B. 深度→体積

UniDepth v2は単眼でもメートル尺度の深度と内参推定を同時に返せるので、一般の投稿写真でもスケール不定性を大幅に低減できます（v2はエッジ改善・高速化・ONNX対応・信頼度出力）。
GitHub

皿・丼・コップは容器幾何（円板/円錐台/円筒）で上限拘束を入れると過大推定抑止に効きます。

既往の単眼深度を用いた体積推定（Food volume estimation / DPF‑Nutrition など）の手法はこの方針と一致。
ACM Digital Library
GitHub
PMC

C. プロンプト設計（Qwen2.5‑VL）

入力（複数画像）

画像1：元写真

画像2：IDごとに輪郭＋番号を描いたオーバーレイ

画像3：深度ヒートマップ

画像4：皿面基準の高さマップ

指示文（要旨）

「各セグメントIDについて {"id": 3, "料理/材料": "...", "性状": "固形/液体/葉物/粒状", "主材料": [{"name": "...", "割合": 0.6}, ...]} を厳格JSONで返して。**推定根拠（短文）と不確実性レベル(0–1)**も付与。IDと論理名の整合を厳守。」

Qwen2.5‑VL はボックス/ポイントの位置合わせと**構造化出力(JSON)**が得意で、複数画像入力も公式チュートリアルに含まれます。
Qwen
Hugging Face

D. 量と栄養の推定

密度：FAO/INFOODSの密度DBをベースにカテゴリ別の代表値＋料理固有の上書き辞書（例：炊飯米、マヨベース、葉物サラダ、スープなど）を持つ。
FAOHome

調理補正：日本の重量変化率（茹で/炒め/焼き…）と、必要に応じUSDA Retention Factorsで微量栄養の保持率を補正。
Food Database
ARS

DBマッピング：優先は日本食品標準成分表2020（八訂）（増補2023に追随）、該当食品なし・ブランド品はUSDA FoodData Central。両者の名称同定辞書（同義・かな/漢字/外来語）を整備。
MEXT
Food Database
FoodData Central

評価プロトコル（現実的なKPI）

分割 IoU / PQ（Food/Non‑food/Utensilを分ける）。

量（g）MAE / MAPE：材料別・料理別に。±20%以内を1段階合格、±10%で上位。

エネルギー（kcal）MAE：盛付け1皿あたり、**±15%**を目標。

アブレーション：幾何のみ / 意味のみ / 融合最適化あり。

データ：FoodSeg103/UEC‑FoodPixで分割、自前で重量実測（キッチンスケール）した小規模評価セット（50–100皿）を用意。
Xiongwei's Homepage
Yanai Lab

運用Tips（精度を押し上げるポイント）

スケール拘束の工夫：
（優）容器径推定→mm/pxに換算　（良）EXIFの焦点距離＋被写体距離ヒューリスティクス　（可）人口統計に基づく標準食器の事前分布。

混在料理（カレー、丼、サラダ）：**入れ子分割（料理→具材）**で “ルー/ご飯/具材” を別IDに。

数え物（唐揚げ/海老）：Qwenの検出→カウントを使い、個数×平均体積で上限/下限拘束。
Qwen

日本語正規化：かな/カナ/漢字の辞書整備。かな読み（ひらがな）を別フィールドに保持。

人手微修正UI：ID統合/分割、容器種別、既知量（例：茶碗一杯=約○g）を1クリックで反映→再最適化。

参考プロンプト（抜粋）

System: あなたは栄養士アシスタントです。画像群（元/マスク/深度/高さ）を見て、各セグメントIDごとに 料理/材料・性状・主材料と割合を厳格JSONで返答しなさい。
User: 画像1〜4を参照。出力は [{id, name_jp, reading, kind, ingredients:[{name_jp, reading, ratio}], note, confidence}]。name_jp/ingredients.name_jp は漢字/かな/カナを含めて返すこと。

量推定ロジック（要旨）

皿平面を点群から推定、各画素の高さ h を算出。

V = Σ h(x,y)·a_pix （負値は0クリップ）。液体は容器リムで満たん率推定（円錐台/円筒モデル）。

m = ρ·V（ρはカテゴリ/材料依存）。

kcal = Σ_i m_i[g] × kcal_i[per g]。

補正：重量変化率・栄養保持率を調理法に応じ適用。
Food Database
ARS

最小出力仕様（あなたの過去の要望に合わせて）
{
  "DBのリスト": [
    {"名前": "ご飯（白飯）", "よみ": "ごはん", "カナ": "ゴハン", "量": 180, "エネルギー_kcal": 302},
    {"名前": "カレー（ルウ）", "よみ": "かれー", "カナ": "カレー", "量": 120, "エネルギー_kcal": 150},
    {"名前": "じゃがいも", "よみ": "じゃがいも", "カナ": "ジャガイモ", "量": 60, "エネルギー_kcal": 46}
  ],
  "合計エネルギー_kcal": 498,
  "不確実性": {"95%CI_kcal": [450, 560]}
}

学習/微調整の実務フロー（Colab/ローカルOK）

分割：SAM 2.1 を FoodSeg103/UEC‑FoodPixで軽くFT（HQ化や食器/影の除外も学習）。
Xiongwei's Homepage
Yanai Lab

深度：UniDepth v2 をそのまま利用（必要なら食卓ドメインで追加調整）。
GitHub

VLM：Qwen2.5‑VL‑7B/72B‑Instructに対し、あなたのID付き画像群→厳格JSONの指示追従LoRAを実施（公開の微調整レポも参考）。
Hugging Face
GitHub

密度/辞書：FAO密度DB＋自前上書き辞書（和食特有：煮物、和え物、出汁類）。
FAOHome

DB連携：**日本食品標準成分表（八訂/増補）を主、USDA FDCを従。名称同定は二段階（正規化→曖昧一致）**で。
MEXT
Food Database
FoodData Central

既知の難所と対応

サラダ/炒飯など混合物：上位材料3–5種の比率推定をVLMに強制、幾何由来総量と比率拘束で配分。

汁物：容器幾何＋液面検出で容量拘束、固形具は別IDで固形密度。

反射/蒸気/ブラー：UniDepthの信頼度で画素重み、不足部は容器拘束へ。

ユーザー毎の盛付け癖：個人事前（茶碗サイズ、よく使う食材密度上書き）を学習。

参考・根拠（主なもの）

SAM 2.1：改良チェックポイント“2.1”と学習/デモ/最適化の公開。
GitHub

SAM 2.1の提供状況（SageMaker JumpStart等）。
Amazon Web Services, Inc.

Qwen2.5‑VL：位置合わせ（bbox/points）と安定JSON、複数画像の利用例。
Qwen
Hugging Face

UniDepth v2：単眼メトリック深度＋内参＋信頼度、v2の改善点。
GitHub

食画像向け分割データ：FoodSeg103 / UEC‑FoodPix Complete。
Xiongwei's Homepage
Yanai Lab

密度表：FAO/INFOODS Density Database（体積→質量変換の基盤）。
FAOHome

調理補正：日本の重量変化率の定義と運用。
Food Database

補助：USDA FoodData Central / FNDDSドキュメント。
FoodData Central

必要なら、この設計をあなたの既存JSON仕様（「DBのリスト」「量」）に完全準拠する推論スクリプト雛形（SAM2.1→UniDepth v2→Qwen2.5‑VL→拘束最適化→JSON）までまとめてお渡しします。

あなた:
A. 分割とデータセット

インスタンス分割は SAM 2.1 を中核に。FoodSeg103（Ingredients/Pixel）とUEC‑FoodPix Complete（料理ピクセル）で微調整 or 合成学習が有効。前者は材料レベル、後者は料理レベルのマスクがあり、組み合わせで階層分割の足場になります。
→このステップを行うプロジェクトをゼロから立ち上げたい。データセットのダウンロード（可能ならコードベースむりならWebブラウザも可能）、各種学習関連パラメータの設定、データ前処理、モデルのロードと学習方法などを含めたプロジェクト実装の曖昧性のない詳細な（コードを含む）プランをフルで出力して。
ChatGPT:
GPT-5 pro に対するリクエスト
Reasoned for 13m 6s

以下は、**「SAM 2.1 を中核に、FoodSeg103（材料レベル）＋UEC‑FoodPix Complete（料理レベル）でインスタンス分割を学習」**するための、曖昧性ゼロの実装計画（ダウンロード〜前処理〜学習〜検証まで、フルコード付き）です。Colab（推奨）／ローカルGPUのどちらでも実行できます。

要点

SAM 2.1は公式リポジトリで学習／微調整コードが公開済み（2024‑09‑30の開発者向けスイートに含まれる）。最低要件は Python ≥3.10、PyTorch ≥2.5.1。チェックポイントの自動DLスクリプトと学習用Hydra設定が同梱されています。
GitHub

FoodSeg103は研究室の配布ページ／GitHubにZIPと解凍パスワード記載あり（LARCdataset9947）。材料のピクセル単位ラベル付き。
GitHub

UEC‑FoodPix Completeは公式ページから直接ダウンロード可。RチャネルにクラスIDが入ったPNGマスク、train.txt／test.txtや category.txt を含む構造が明示されています（非商用研究目的に限定）。
Yanai Lab

SAM2 の学習コードは画像（SA‑1B 形式）／動画（DAVIS系）を両対応。画像学習ではSA‑1B風（画像ごとJSON＋RLE）を例として実装されているため、FoodSeg103/UECのセマンティックマスク→連結成分ごとのオブジェクトマスクに分解してSA‑1B形式へ正規化し、そのまま学習に使用します。
Hugging Face
GitHub

0) 成果物（プロジェクト構成）
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

1) 環境構築（Colab / ローカルGPU）
1‑A. 依存関係とSAM2.1の導入
# 00_setup_env.sh
set -eux

# (Colabなら不要) 新規venv or conda推奨
# conda create -n foodsammix python=3.10 -y && conda activate foodsammix

# pytorch 2.5.1+ を環境に合わせて (以下は例。CUDA版は https://pytorch.org/ を参照)
pip install --upgrade pip
# GPU環境に合わせてインストールしてください（例: cu124）
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


SAM 2.1 の学習コード・チェックポイント・**要件（Python3.10 / Torch2.5.1+）**は公式READMEに明記されています。
GitHub

2) データセットの取得

ライセンス注意

UEC‑FoodPix/Completeは非商用研究のみ利用可。研究用途であることを確認してください。
Yanai Lab

FoodSeg103は研究目的配布。公式リポの指示に従い、ZIPとパスワードを用いて取得します。
GitHub

2‑A. FoodSeg103
# 01_download_foodseg103.sh
set -eux
cd "$(git rev-parse --show-toplevel)"

mkdir -p data/FoodSeg103 && cd data/FoodSeg103
# 公式配布ZIP（GitHub READMEにURLとパスワードが記載）
# URL例: https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip
# パスワード: LARCdataset9947
curl -L -o FoodSeg103.zip "https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip"
unzip -P LARCdataset9947 FoodSeg103.zip


GitHubのREADMEに「FoodSeg103.zip を ./data/FoodSeg103/ に展開し、パスワードLARCdataset9947」と明記。
GitHub

データは材料クラスのピクセルラベルを含みます（Dataset概要）。
Xiongwei's Homepage

2‑B. UEC‑FoodPix Complete
# 02_download_uecfoodpix.sh
set -eux
cd "$(git rev-parse --show-toplevel)"

mkdir -p data && cd data
# 公式DLリンク (サイトに直リンクが記載)
# https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar
curl -L -o UECFOODPIXCOMPLETE.tar "https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar"
tar -xf UECFOODPIXCOMPLETE.tar


サイト上にDownloadsの直リンク（.tar）とディレクトリ構造（img/, mask/, train.txt, test.txt, category.txt）が明示。マスクはRチャネルにクラスID。
Yanai Lab

3) SA‑1B 風（画像ごとJSON＋RLE）への正規化

SAM2.1 の画像学習は、SA‑1B 風のアノテーション（画像ごとJSON／RLEマスク配列）のサンプル実装が用意されています。ここでは FoodSeg103/UEC のセマンティックマスクを連結成分ごとに分解して、**クラス非依存の“オブジェクトマスク群”**に変換し、SA‑1B と同じRLE構造で保存します（annotations/<image_basename>.json）。
Hugging Face
GitHub

ポイント：FoodSeg103/UECは「ピクセルごとにクラスID」です。各クラス内の連結成分を1インスタンスと見なし、背景（ID=0）は除外します。UECはRチャネルの画素値がIDなのでそれを読みます。
Yanai Lab

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

3‑A. 学習・検証スプリットの作成
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

4) 画像サイズの正規化（任意／推奨）

Roboflowの実地ガイドでは、学習設定に合わせて 1024×1024 にストレッチしたデータで微調整する例が示されています。SAM2.1の学習設定によっては入力解像度を固定する前提があるため、安定運用のために事前リサイズを選ぶ場合は以下を実行してください。
Roboflow Blog

# scripts/13_optional_resize_1024.py
import os, glob
from PIL import Image
from tqdm import tqdm

IN_DIR = "data/foodmix_sa1b/images"
OUT_DIR = IN_DIR  # 上書き (別ディレクトリを使いたい場合は変更)

for p in tqdm(glob.glob(os.path.join(IN_DIR, "*.*"))):
    im = Image.open(p).convert("RGB").resize((1024,1024), Image.BILINEAR)
    im.save(p)


公式SAM2学習コードは**Hydra設定で変換（transforms）**を持ち、SA‑1Bや動画データの例が付属しています（training/dataset/と設定YAML）。本手順では、事前リサイズで設定側の依存を減らしています。
Hugging Face

5) SAM 2.1 の学習設定（Hydra YAML）

SAM2.1 リポのMOSE微調整の設定YAMLを複製してパスだけ差し替えるのが最も確実です（configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml を ..._foodmix_finetune.yaml にコピー）。SAM2.1 公式の training/train.py は画像データに対して SA1BRawDataset を例示しています。以下は最小限の置換例です：
Hugging Face

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


学習の起動コマンド（公式READMEの手順通り）：

cd external/sam2
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml \
  --use-cluster 0 --num-gpus 1


（A100等マルチGPUなら --num-gpus 8／SLURMオプションも可）
Hugging Face

参考：RoboflowのSAM‑2.1微調整ガイドも、40エポック・1024入力・training/train.py 実行などの実例を示しています。
Roboflow Blog

6) 学習確認・可視化・簡易評価
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

7) 実運用パラメータ（推奨初期値）

バックボーン：sam2.1_hiera_base_plus.pt（精度と速度のバランス）。sam2.1_hiera_large.pt は精度重視。
GitHub

入力解像度：まずは 1024×1024 固定（前処理リサイズ）。設定で可変にする場合は transforms を既定から流用。
Roboflow Blog

バッチサイズ：単GPU/T4なら 1、A100なら 2–4。

エポック：40（初期探索の目安）。
Roboflow Blog

最適化：AdamW、lr=1e-4（必要に応じ 1e-5〜3e-4 で格子探索）。

精度モード：A100等は bf16、T4系は fp32（設定YAMLの precision）。
Hugging Face

最大物体数：max_num_objects=50（盛付けの細粒度を考慮し高めに）

8) よくある落とし穴と対策

学習用JSONのRLE
SA‑1B形式のRLEはFortran orderでエンコード（pycocotools.encode(np.asfortranarray(...))）すること。
GitHub

UECのマスク解釈
RチャネルのみがクラスID。RGB全体で扱わない。
Yanai Lab

ライセンス
UECは非商用研究限定。成果公開時は引用も忘れずに（サイトにBibTeX例あり）。
Yanai Lab

学習コードの場所
SAM2.1 リポの training/ にデータセットクラス・Trainer・optimizer 等があり、training/train.py で起動。Hydra設定のパスだけ正しく。
Hugging Face

入出力回り
画像名と annotations/<stem>.json の整合、train.txt/val.txt の改行・拡張子に注意。

9) 追加メモ（拡張）

カテゴリ情報を保持したい場合：JSONの各annotationに独自キー（例 {"category": <id>}）を足しておいても、SAM2学習のマスク品質には影響しません（クラス非依存学習）。

評価の厳密化：COCO‑style AP（IoUしきい値系列）やBoundary F‑scoreの計測ツールを追加実装可。

学習スキーマ：材料レベル（FoodSeg103）と料理レベル（UEC）を混在させることで、「食材／具材の細かい境界」と「料理輪郭」の両方に強いモデルを狙えます。

10) 実行手順まとめ
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

参考・根拠

SAM 2.1 公式：学習コード・チェックポイント・要件、SAM 2.1 の公開（2024‑09‑30、Training/README 参照）、使用例・コンフィグパス。
GitHub

SA‑1B 形式：画像ごとJSON＋RLEの構造（image／annotations）、RLEはCOCO形式。
GitHub

FoodSeg103：公式HPとGitHub。ZIP配布とパスワード、材料ピクセルラベル。
Xiongwei's Homepage
GitHub

UEC‑FoodPix Complete：ダウンロードとファイル構造、Rチャネルがラベル、利用は非商用研究目的。
Yanai Lab

実地ガイド：SAM‑2.1 微調整手順（1024入力、40エポック、training/train.pyで学習）。
Roboflow Blog

必要でしたら、このプランをColabノートブック一式（上記スクリプトをセル化、ログ／TensorBoard表示込み）に再構成してお渡しできます。どのGPU（T4/A100）で回すか、所要時間に応じてバッチサイズ／精度設定の推奨値も具体化します。