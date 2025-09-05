# SAM2学習時のメモリ最適化クエリ

## 問題状況
- SAM2.1の学習中にメモリ使用量が6GB→42GBに急激に増加
- バッチサイズは1でも発生する問題
- PCが重くなり途中でクラッシュする

## 調査すべき点
- SAM2公式実装のメモリ効率化手法
- gradient_checkpointingの設定
- mixed precisionの最適化
- データローダーのメモリリーク対策
- torch.cuda.empty_cache()の適切なタイミング

SAM2.1のファインチューニング時にメモリ使用量が学習の進行と共に急激に増加する問題について、以下の観点から解決策を調査したい：

1. SAM2公式実装でのgradient checkpointingやmixed precisionの最適設定
2. 学習中のメモリリークを防ぐためのtorch.cuda.empty_cache()の使用タイミング
3. データローダーのnum_workersやpin_memoryの最適化
4. バッチサイズ1でも発生するメモリ増加の原因と対策
5. 長時間学習でもメモリ使用量を安定させる実装方法

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。