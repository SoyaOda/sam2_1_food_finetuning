# SAM2.1 メモリ最適化クエリ

## 現在の問題
SAM2.1 の fine-tuning 中に GPU メモリ使用量が step 250 付近で 98.9% まで上昇し、システムクラッシュのリスクがある。現在の設定でも memory leak が発生している可能性がある。

## 調査したい内容
1. SAM2.1 公式実装でのメモリ効率的な学習設定とベストプラクティス
2. PyTorch と DDP を使った大規模モデル学習でのメモリリーク回避手法
3. Segment Anything Model の学習で推奨されるバッチサイズ、解像度、workers 設定
4. gradient checkpointing や mixed precision の適切な設定方法
5. WSL2 環境での CUDA メモリ管理の注意点と最適化手法
6. SAM2 学習中のメモリスパイクを防ぐための前処理やデータローダー設定
7. メモリ使用量を大幅に削減できる代替的なアプローチ（LoRA、QLoRA等）

## 現在の設定
- batch_size: 1
- resolution: 1024
- num_train_workers: 2
- max_num_objects: 20
- mixed precision 有効
- gradient checkpointing 無効（互換性問題のため）

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。