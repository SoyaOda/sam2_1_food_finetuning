# FoodSeg103とUECFoodPixデータのSA-1B形式変換に関する技術調査クエリ

## 調査内容

SAM2.1の学習用にFoodSeg103（食材セグメンテーション）とUECFoodPixComplete（料理セグメンテーション）のセマンティックマスクデータを、Meta社のSA-1B形式のインスタンスセグメンテーションデータに変換する際の技術的詳細について調査したい。

### 具体的な調査項目

1. **FoodSeg103のデータ構造と読み込み方法**
   - 実際のマスクファイル形式（PNG？）
   - クラスIDのエンコーディング方式
   - 画像とマスクファイルの対応関係
   - 公式で推奨されている読み込み方法

2. **UECFoodPixCompleteのデータ構造**
   - マスクファイルでのRチャネルのクラスID格納方式
   - train.txt, test.txt, category.txtの具体的な形式
   - 実際のファイル構造とパス

3. **SA-1B形式のRLEエンコーディング**
   - pycocotoolsを使ったFortran orderでのRLEエンコーディングの正確な実装
   - Meta社の公式SA-1Bデータセットの構造
   - annotations/<image_basename>.jsonの正確な形式

4. **セマンティックマスクからインスタンスマスクへの変換**
   - scikit-imageのlabel()関数を使った連結成分分析の最適な実装
   - 小さすぎるインスタンスのフィルタリング基準
   - bbox計算とarea計算の正確な方法

5. **SAM2.1の学習データローダーとの互換性**
   - SA1BRawDatasetクラスが期待するデータ形式
   - training/dataset/vos_raw_dataset.pyでの実装詳細
   - Hydra設定での適切なパス指定方法

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。