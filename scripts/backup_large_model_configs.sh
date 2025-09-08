#!/bin/bash
# backup_large_model_configs.sh - SAM2.1 Largeモデル設定ファイルのバックアップスクリプト
set -e

echo "==============================================="
echo "SAM2.1 Largeモデル設定ファイルバックアップ"
echo "==============================================="
echo ""

# プロジェクトルートの確認
if [ ! -f "CLAUDE.md" ]; then
    echo "エラー: プロジェクトルートで実行してください"
    exit 1
fi

# バックアップディレクトリの作成
BACKUP_DIR="backup/large_model_configs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_SUBDIR="$BACKUP_DIR/$TIMESTAMP"

echo "バックアップディレクトリを作成: $BACKUP_SUBDIR"
mkdir -p "$BACKUP_SUBDIR"

echo ""
echo "=== プロジェクト内設定ファイルのバックアップ ==="

# メイン設定ファイルのバックアップ
MAIN_CONFIG_DIR="configs/sam2.1_training"
if [ -d "$MAIN_CONFIG_DIR" ]; then
    echo "メイン設定ファイルをコピー中..."
    cp "$MAIN_CONFIG_DIR"/sam2.1_hiera_l_*.yaml "$BACKUP_SUBDIR/" 2>/dev/null || echo "Largeモデル設定ファイルが見つかりません"
    
    # 参照用にBase+設定もバックアップ
    cp "$MAIN_CONFIG_DIR"/sam2.1_hiera_b+_*.yaml "$BACKUP_SUBDIR/" 2>/dev/null || echo "Base+設定ファイルが見つかりません"
    
    echo "✓ メイン設定ファイルのバックアップ完了"
else
    echo "警告: $MAIN_CONFIG_DIR が見つかりません"
fi

echo ""
echo "=== SAM2リポジトリ内設定ファイルのバックアップ ==="

# SAM2リポジトリ内の設定ファイルのバックアップ
SAM2_CONFIG_DIR="external/sam2/sam2/configs/sam2.1_training"
if [ -d "$SAM2_CONFIG_DIR" ]; then
    echo "SAM2リポジトリ内設定ファイルをコピー中..."
    
    # Largeモデル設定ファイル
    cp "$SAM2_CONFIG_DIR"/sam2.1_hiera_l_*.yaml "$BACKUP_SUBDIR/" 2>/dev/null || echo "SAM2リポジトリ内Largeモデル設定が見つかりません"
    
    # 参照用にBase+設定もバックアップ
    cp "$SAM2_CONFIG_DIR"/sam2.1_hiera_b+_foodmix_*.yaml "$BACKUP_SUBDIR/" 2>/dev/null || echo "SAM2リポジトリ内Base+設定が見つかりません"
    
    echo "✓ SAM2リポジトリ内設定ファイルのバックアップ完了"
else
    echo "警告: $SAM2_CONFIG_DIR が見つかりません"
fi

echo ""
echo "=== スクリプトファイルのバックアップ ==="

# 実行スクリプトのバックアップ
if [ -f "scripts/20_train_food_sam2_large.sh" ]; then
    cp scripts/20_train_food_sam2_large.sh "$BACKUP_SUBDIR/"
    echo "✓ Largeモデル学習スクリプトのバックアップ完了"
fi

if [ -f "scripts/train_with_monitoring.sh" ]; then
    cp scripts/train_with_monitoring.sh "$BACKUP_SUBDIR/"
    echo "✓ 監視付き学習スクリプトのバックアップ完了"
fi

echo ""
echo "=== ドキュメントファイルのバックアップ ==="

# ドキュメントのバックアップ
DOC_FILES=(
    "docs/large_model_optimization_guide.md"
    "docs/large_model_config_mapping.md"
    "docs/large_model_file_placement.md"
    "md_files/large_model_shift_spec.md"
)

for doc_file in "${DOC_FILES[@]}"; do
    if [ -f "$doc_file" ]; then
        cp "$doc_file" "$BACKUP_SUBDIR/"
        echo "✓ $(basename "$doc_file") のバックアップ完了"
    fi
done

echo ""
echo "=== バックアップ内容の確認 ==="

echo "バックアップされたファイル:"
ls -la "$BACKUP_SUBDIR"

BACKUP_COUNT=$(ls -1 "$BACKUP_SUBDIR" | wc -l)
echo ""
echo "バックアップファイル数: $BACKUP_COUNT"

# 最新バックアップへのシンボリックリンク作成
LATEST_LINK="$BACKUP_DIR/latest"
if [ -L "$LATEST_LINK" ]; then
    rm "$LATEST_LINK"
fi
ln -s "$TIMESTAMP" "$LATEST_LINK"
echo "最新バックアップリンク作成: $LATEST_LINK -> $TIMESTAMP"

echo ""
echo "=== Git管理への追加オプション ==="

read -p "バックアップをGitリポジトリに追加しますか？ (y/N): " ADD_TO_GIT
if [ "$ADD_TO_GIT" = "y" ] || [ "$ADD_TO_GIT" = "Y" ]; then
    git add "$BACKUP_SUBDIR"
    echo "✓ バックアップファイルをGitステージングエリアに追加しました"
    echo "コミットするには: git commit -m \"backup: SAM2.1 Large model configs ($TIMESTAMP)\""
else
    echo "バックアップファイルはローカルに保存されました (Gitには追加されませんでした)"
fi

echo ""
echo "=== 復元方法 ==="
echo "サブモジュール更新などでファイルが失われた場合の復元方法:"
echo ""
echo "# 最新バックアップからの復元"
echo "cp $BACKUP_DIR/latest/*.yaml configs/sam2.1_training/"
echo "cp $BACKUP_DIR/latest/*l_foodmix*.yaml external/sam2/sam2/configs/sam2.1_training/"
echo ""
echo "# 特定日時からの復元"
echo "cp $BACKUP_SUBDIR/*.yaml configs/sam2.1_training/"
echo "cp $BACKUP_SUBDIR/*l_foodmix*.yaml external/sam2/sam2/configs/sam2.1_training/"

echo ""
echo "==============================================="
echo "バックアップ完了"
echo "==============================================="
echo "バックアップ場所: $BACKUP_SUBDIR"
echo "最新リンク: $LATEST_LINK"