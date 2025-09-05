# SAM2.1 Food Finetuning Project

## プロジェクト概要
SAM 2.1を中核として、FoodSeg103（材料レベル）とUEC-FoodPix Complete（料理レベル）データセットを使用してインスタンス分割モデルを学習するプロジェクト。

## 技術スタック
- **Python**: 3.10+
- **PyTorch**: 2.5.1+
- **SAM 2.1**: Meta公式のSegment Anything Model v2.1
- **依存ライブラリ**: opencv-python, numpy, pycocotools, scikit-image, pillow, tqdm, hydra-core, omegaconf, matplotlib

## データセット
1. **FoodSeg103**: 材料レベルのピクセル単位ラベル付きデータセット
   - パスワード: LARCdataset9947
   - 研究目的配布

2. **UEC-FoodPix Complete**: 料理レベルのデータセット
   - RチャネルにクラスID
   - 非商用研究目的限定

## プロジェクト構成
```
foodsammix/
├─ external/sam2/            # SAM 2.1 公式リポジトリ
├─ data/                     # データセット格納
│   ├─ FoodSeg103/
│   ├─ UECFOODPIXCOMPLETE/
│   └─ foodmix_sa1b/        # SA-1B形式に正規化したデータ
├─ scripts/                  # 実装スクリプト
└─ configs/                  # 学習設定YAML
```

## 実装ステップ
1. 環境構築とSAM2.1セットアップ
2. データセットダウンロード（FoodSeg103, UEC-FoodPix）
3. SA-1B形式へのデータ正規化
4. 画像リサイズ（1024x1024）
5. 学習設定YAMLの作成
6. モデル学習実行
7. 評価・可視化

## 重要な実装ポイント
- RLEはFortran orderでエンコード
- UECマスクはRチャネルのみ使用
- セマンティックマスクを連結成分ごとのインスタンスマスクに分解
- SA-1B互換のJSON形式でアノテーション保存