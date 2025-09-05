#!/bin/bash
# 02_download_uecfoodpix.sh - UEC-FoodPix Completeデータセットダウンロードスクリプト
set -eux

echo "UEC-FoodPix Completeデータセットのダウンロードを開始します..."
echo "注意: このデータセットは非商用研究目的でのみ使用可能です"

# プロジェクトルートを取得
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# データディレクトリ作成
mkdir -p data
cd data

# ダウンロード済みチェック
if [ -d "UECFOODPIXCOMPLETE" ]; then
    echo "UECFOODPIXCOMPLETEディレクトリが既に存在します"
    
    # データの存在確認
    if [ -d "UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE" ] || [ -d "UECFOODPIXCOMPLETE/train" ]; then
        echo "データが既に存在するようです"
        echo "再ダウンロードする場合は、data/UECFOODPIXCOMPLETEディレクトリを削除してください"
        exit 0
    fi
fi

# TARファイルのダウンロード
UEC_URL="https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar"
TAR_FILE="UECFOODPIXCOMPLETE.tar"

if [ ! -f "$TAR_FILE" ]; then
    echo "UECFOODPIXCOMPLETE.tarをダウンロード中..."
    echo "URL: $UEC_URL"
    echo "このファイルは約1.5GBあります。ダウンロードに時間がかかる場合があります..."
    
    # curlでダウンロード（プログレスバー表示）
    if command -v curl &> /dev/null; then
        curl -L -o "$TAR_FILE" "$UEC_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$TAR_FILE" "$UEC_URL"
    else
        echo "エラー: curlまたはwgetが必要です"
        exit 1
    fi
    
    # ダウンロード確認
    if [ ! -f "$TAR_FILE" ]; then
        echo "エラー: ダウンロードに失敗しました"
        echo "手動でダウンロードしてください: $UEC_URL"
        exit 1
    fi
    
    echo "ダウンロード完了: $TAR_FILE"
else
    echo "$TAR_FILE が既に存在します"
fi

# ファイルサイズ確認（約1.5GB）
FILE_SIZE=$(stat -c%s "$TAR_FILE" 2>/dev/null || stat -f%z "$TAR_FILE" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt 100000000 ]; then
    echo "警告: ファイルサイズが小さすぎます (${FILE_SIZE} bytes)"
    echo "ダウンロードが不完全な可能性があります"
    rm -f "$TAR_FILE"
    echo "再度実行してください"
    exit 1
fi

# TARファイルの解凍
echo "TARファイルを解凍中..."

# tarコマンドの存在確認
if ! command -v tar &> /dev/null; then
    echo "エラー: tarコマンドが見つかりません"
    exit 1
fi

# 解凍
tar -xf "$TAR_FILE" || {
    echo "エラー: 解凍に失敗しました"
    echo "手動で解凍してください: tar -xf $TAR_FILE"
    exit 1
}

echo "解凍完了"

# ディレクトリ構造の確認
echo ""
echo "=== ディレクトリ構造 ==="
ls -la UECFOODPIXCOMPLETE/ 2>/dev/null || ls -la

# データ構造の詳細確認
echo ""
echo "=== データ確認 ==="

# 期待されるディレクトリ構造を確認
BASE_DIR=""
if [ -d "UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE" ]; then
    BASE_DIR="UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE"
elif [ -d "UECFOODPIXCOMPLETE/data/UECFOODPIXCOMPLETE" ]; then
    BASE_DIR="UECFOODPIXCOMPLETE/data/UECFOODPIXCOMPLETE"
elif [ -d "data/UECFoodPIXCOMPLETE" ]; then
    BASE_DIR="data/UECFoodPIXCOMPLETE"
elif [ -d "UECFoodPIXCOMPLETE" ]; then
    BASE_DIR="UECFoodPIXCOMPLETE"
fi

if [ -n "$BASE_DIR" ]; then
    echo "ベースディレクトリ: $BASE_DIR"
    
    # trainデータの確認
    if [ -d "$BASE_DIR/train" ]; then
        echo ""
        echo "訓練データ:"
        if [ -d "$BASE_DIR/train/img" ]; then
            TRAIN_IMG_COUNT=$(find "$BASE_DIR/train/img" -name "*.jpg" | wc -l)
            echo "  画像数 (train/img): $TRAIN_IMG_COUNT"
        fi
        if [ -d "$BASE_DIR/train/mask" ]; then
            TRAIN_MASK_COUNT=$(find "$BASE_DIR/train/mask" -name "*.png" | wc -l)
            echo "  マスク数 (train/mask): $TRAIN_MASK_COUNT"
        fi
    fi
    
    # testデータの確認
    if [ -d "$BASE_DIR/test" ]; then
        echo ""
        echo "テストデータ:"
        if [ -d "$BASE_DIR/test/img" ]; then
            TEST_IMG_COUNT=$(find "$BASE_DIR/test/img" -name "*.jpg" | wc -l)
            echo "  画像数 (test/img): $TEST_IMG_COUNT"
        fi
        if [ -d "$BASE_DIR/test/mask" ]; then
            TEST_MASK_COUNT=$(find "$BASE_DIR/test/mask" -name "*.png" | wc -l)
            echo "  マスク数 (test/mask): $TEST_MASK_COUNT"
        fi
    fi
    
    # メタデータファイルの確認
    echo ""
    echo "メタデータファイル:"
    [ -f "$BASE_DIR/train.txt" ] && echo "  train.txt: 存在"
    [ -f "$BASE_DIR/test.txt" ] && echo "  test.txt: 存在"
    [ -f "$BASE_DIR/category.txt" ] && echo "  category.txt: 存在"
    
    # category.txtの内容を一部表示
    if [ -f "$BASE_DIR/category.txt" ]; then
        echo ""
        echo "カテゴリ例（最初の5行）:"
        head -5 "$BASE_DIR/category.txt" | sed 's/^/  /'
    fi
    
else
    echo "警告: 期待されるディレクトリ構造が見つかりません"
    echo "手動で確認してください:"
    find . -maxdepth 3 -type d -name "*img*" -o -name "*mask*" | head -20
fi

cd "$PROJECT_ROOT"

echo ""
echo "=== 重要な注意事項 ==="
echo "UEC-FoodPix Completeのマスク画像について:"
echo "- RチャネルにクラスIDが格納されています"
echo "- 背景は0、食材クラスは1以上の値です"
echo "- RGBではなく、Rチャネルのみを使用してください"

echo ""
echo "UEC-FoodPix Completeデータセットのダウンロードが完了しました！"
echo "データ場所: data/UECFOODPIXCOMPLETE/"
echo ""
echo "次のステップ:"
echo "1. python scripts/10_prepare_foodseg103.py - FoodSeg103の前処理"
echo "2. python scripts/11_prepare_uecfoodpix.py - UEC-FoodPixの前処理"