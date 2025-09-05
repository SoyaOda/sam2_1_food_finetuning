#!/bin/bash
# 01_download_foodseg103.sh - FoodSeg103データセットダウンロードスクリプト
set -eux

echo "FoodSeg103データセットのダウンロードを開始します..."
echo "注意: このデータセットは研究目的でのみ使用可能です"

# プロジェクトルートを取得
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# データディレクトリ作成
mkdir -p data/FoodSeg103
cd data/FoodSeg103

# ダウンロード済みチェック
if [ -d "FoodSeg103" ] || [ -d "Images" ] || [ -d "images" ]; then
    echo "FoodSeg103データが既に存在するようです"
    echo "再ダウンロードする場合は、data/FoodSeg103ディレクトリを削除してください"
    exit 0
fi

# ZIPファイルのダウンロード
FOODSEG_URL="https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip"
ZIP_FILE="FoodSeg103.zip"
PASSWORD="LARCdataset9947"

if [ ! -f "$ZIP_FILE" ]; then
    echo "FoodSeg103.zipをダウンロード中..."
    echo "URL: $FOODSEG_URL"
    
    # curlでダウンロード（プログレスバー表示）
    if command -v curl &> /dev/null; then
        curl -L -o "$ZIP_FILE" "$FOODSEG_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$ZIP_FILE" "$FOODSEG_URL"
    else
        echo "エラー: curlまたはwgetが必要です"
        exit 1
    fi
    
    # ダウンロード確認
    if [ ! -f "$ZIP_FILE" ]; then
        echo "エラー: ダウンロードに失敗しました"
        echo "手動でダウンロードしてください: $FOODSEG_URL"
        exit 1
    fi
    
    echo "ダウンロード完了: $ZIP_FILE"
else
    echo "$ZIP_FILE が既に存在します"
fi

# ファイルサイズ確認
FILE_SIZE=$(stat -c%s "$ZIP_FILE" 2>/dev/null || stat -f%z "$ZIP_FILE" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "警告: ファイルサイズが小さすぎます (${FILE_SIZE} bytes)"
    echo "ダウンロードが不完全な可能性があります"
    rm -f "$ZIP_FILE"
    echo "再度実行してください"
    exit 1
fi

# ZIPファイルの解凍
echo "ZIPファイルを解凍中..."
echo "パスワード: $PASSWORD"

# unzipコマンドの存在確認
if ! command -v unzip &> /dev/null; then
    echo "エラー: unzipコマンドが見つかりません"
    echo "インストールしてください: sudo apt-get install unzip"
    exit 1
fi

# パスワード付きZIPの解凍
unzip -P "$PASSWORD" "$ZIP_FILE" || {
    echo "エラー: 解凍に失敗しました"
    echo "パスワードが正しいか確認してください: $PASSWORD"
    echo "または手動で解凍してください: unzip -P $PASSWORD $ZIP_FILE"
    exit 1
}

echo "解凍完了"

# ディレクトリ構造の確認
echo ""
echo "=== ディレクトリ構造 ==="
ls -la

# 画像とマスクの確認
echo ""
echo "=== データ確認 ==="

# Imagesディレクトリの確認（大文字小文字の両方をチェック）
IMG_DIR=""
if [ -d "Images" ]; then
    IMG_DIR="Images"
elif [ -d "images" ]; then
    IMG_DIR="images"
elif [ -d "FoodSeg103/Images" ]; then
    IMG_DIR="FoodSeg103/Images"
elif [ -d "FoodSeg103/images" ]; then
    IMG_DIR="FoodSeg103/images"
fi

MASK_DIR=""
if [ -d "Annotations" ]; then
    MASK_DIR="Annotations"
elif [ -d "annotations" ]; then
    MASK_DIR="annotations"
elif [ -d "Masks" ]; then
    MASK_DIR="Masks"
elif [ -d "masks" ]; then
    MASK_DIR="masks"
elif [ -d "FoodSeg103/Annotations" ]; then
    MASK_DIR="FoodSeg103/Annotations"
elif [ -d "FoodSeg103/annotations" ]; then
    MASK_DIR="FoodSeg103/annotations"
fi

if [ -n "$IMG_DIR" ]; then
    echo "画像ディレクトリ: $IMG_DIR"
    IMG_COUNT=$(find "$IMG_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    echo "画像数: $IMG_COUNT"
else
    echo "警告: 画像ディレクトリが見つかりません"
fi

if [ -n "$MASK_DIR" ]; then
    echo "マスクディレクトリ: $MASK_DIR"
    MASK_COUNT=$(find "$MASK_DIR" -type f -name "*.png" | wc -l)
    echo "マスク数: $MASK_COUNT"
else
    echo "警告: マスクディレクトリが見つかりません"
fi

# メタデータファイルの確認
if [ -f "category.txt" ] || [ -f "FoodSeg103/category.txt" ]; then
    echo "category.txtが見つかりました"
fi

cd "$PROJECT_ROOT"

echo ""
echo "FoodSeg103データセットのダウンロードが完了しました！"
echo "データ場所: data/FoodSeg103/"
echo ""
echo "次のステップ:"
echo "1. bash scripts/02_download_uecfoodpix.sh - UEC-FoodPixデータセットのダウンロード"
echo "2. python scripts/10_prepare_foodseg103.py - FoodSeg103の前処理"