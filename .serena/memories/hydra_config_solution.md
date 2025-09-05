# SAM2.1 Hydra設定パス問題の解決方法

## 最も簡単な解決策: --config-dirオプションを使用

```bash
cd /home/soya/sam2_1_food_finetuning/external/sam2

python training/train.py \
    --config-dir sam2/configs \
    --config-name sam2.1_training/sam2.1_hiera_b+_foodmix_finetune \
    --use-cluster 0 --num-gpus 1
```

### なぜこれが機能するか
- `--config-dir sam2/configs`: Hydraに対して設定ファイルを探すルートディレクトリを明示的に指定
- `--config-name`: config-dirからの相対パス（拡張子.yamlを除く）を指定
- これにより`pkg://sam2`というパッケージ内部の検索パスが上書きされ、ローカルファイルシステムを直接参照

## 根本原因
1. `initialize_config_module("sam2")`はインストールされたパッケージ内を探す
2. 新しく追加したYAMLファイルはパッケージの一部として認識されていない
3. Hydraの規約では`conf`ディレクトリを探すが、SAM2は`configs`を使用

## 代替解決策（より複雑）
1. `sam2/configs`を`sam2/conf`にリネーム
2. setup.pyでpackage_dataに.yamlファイルを含める
3. pip install -e .で再インストール

推奨: --config-dirオプションの使用が最も簡単で確実