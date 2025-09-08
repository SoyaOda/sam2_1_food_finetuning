#!/bin/bash
# check_large_model_setup.sh - SAM2.1 Large モデル環境確認スクリプト
set -e

echo "==============================================="
echo "SAM2.1 Hiera-Large モデル環境確認"
echo "==============================================="
echo ""

# カラーコード定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# チェック関数
check_file() {
    local file_path="$1"
    local description="$2"
    
    if [ -f "$file_path" ]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        return 1
    fi
}

check_directory() {
    local dir_path="$1"
    local description="$2"
    
    if [ -d "$dir_path" ]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        return 1
    fi
}

# プロジェクトルートの確認
if [ ! -f "CLAUDE.md" ]; then
    echo -e "${RED}エラー: プロジェクトルートで実行してください${NC}"
    exit 1
fi

echo -e "${BLUE}1. チェックポイントファイル確認${NC}"
echo "----------------------------------------"
CHECKPOINT_OK=0
if check_file "external/sam2/checkpoints/sam2.1_hiera_large.pt" "Largeモデル チェックポイント"; then
    CHECKPOINT_SIZE=$(ls -lh external/sam2/checkpoints/sam2.1_hiera_large.pt | awk '{print $5}')
    echo "   サイズ: $CHECKPOINT_SIZE"
    CHECKPOINT_OK=1
else
    echo -e "   ${YELLOW}対処法: cd external/sam2 && bash checkpoints/download_ckpts.sh${NC}"
fi

if check_file "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt" "Base+モデル チェックポイント (参照用)"; then
    BASEPLUS_SIZE=$(ls -lh external/sam2/checkpoints/sam2.1_hiera_base_plus.pt | awk '{print $5}')
    echo "   サイズ: $BASEPLUS_SIZE"
fi
echo ""

echo -e "${BLUE}2. モデル設定ファイル確認${NC}"
echo "----------------------------------------"
MODEL_CONFIG_OK=0
if check_file "external/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" "Largeモデル 公式設定"; then
    MODEL_CONFIG_OK=1
fi
check_file "external/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml" "Base+モデル 公式設定 (参照用)"
echo ""

echo -e "${BLUE}3. 学習設定ファイル確認 - メイン設定${NC}"
echo "----------------------------------------"
MAIN_CONFIG_COUNT=0

if check_file "configs/sam2.1_training/sam2.1_hiera_l_foodmix_finetune.yaml" "標準ファインチューニング設定"; then
    ((MAIN_CONFIG_COUNT++))
fi

if check_file "configs/sam2.1_training/sam2.1_hiera_l_foodmix_simple.yaml" "シンプル学習設定"; then
    ((MAIN_CONFIG_COUNT++))
fi

# Base+版も確認
check_file "configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml" "Base+ ファインチューニング設定 (参照用)"
check_file "configs/sam2.1_training/sam2.1_hiera_b+_foodmix_simple.yaml" "Base+ シンプル設定 (参照用)"
echo ""

echo -e "${BLUE}4. 学習設定ファイル確認 - 詳細設定${NC}"
echo "----------------------------------------"
DETAIL_CONFIG_COUNT=0

if check_file "external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_optimized.yaml" "メモリ最適化版設定"; then
    ((DETAIL_CONFIG_COUNT++))
fi

if check_file "external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_test.yaml" "1エポック テスト設定"; then
    ((DETAIL_CONFIG_COUNT++))
fi

if check_file "external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_l_foodmix_40epochs.yaml" "40エポック フル学習設定"; then
    ((DETAIL_CONFIG_COUNT++))
fi

# Base+版も確認
check_file "external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_optimized.yaml" "Base+ メモリ最適化版 (参照用)"
echo ""

echo -e "${BLUE}5. 実行スクリプト確認${NC}"
echo "----------------------------------------"
SCRIPT_OK=0
if check_file "scripts/20_train_food_sam2_large.sh" "Largeモデル専用学習スクリプト"; then
    if [ -x "scripts/20_train_food_sam2_large.sh" ]; then
        echo -e "   ${GREEN}実行権限: あり${NC}"
        SCRIPT_OK=1
    else
        echo -e "   ${YELLOW}実行権限: なし (chmod +x scripts/20_train_food_sam2_large.sh で修正)${NC}"
    fi
fi

check_file "scripts/20_train_food_sam2.sh" "Base+モデル学習スクリプト (参照用)"
check_file "scripts/train_with_monitoring.sh" "監視付き学習スクリプト"
echo ""

echo -e "${BLUE}6. ドキュメント確認${NC}"
echo "----------------------------------------"
DOC_COUNT=0

if check_file "docs/large_model_optimization_guide.md" "最適化ガイド"; then
    ((DOC_COUNT++))
fi

if check_file "docs/large_model_config_mapping.md" "設定ファイル対応表"; then
    ((DOC_COUNT++))
fi

if check_file "docs/large_model_file_placement.md" "ファイル配置ガイド"; then
    ((DOC_COUNT++))
fi

check_file "md_files/large_model_shift_spec.md" "移行仕様書"
echo ""

echo -e "${BLUE}7. データセット確認${NC}"
echo "----------------------------------------"
DATA_OK=0
if check_directory "data/foodmix_sa1b/images" "学習画像ディレクトリ" && \
   check_directory "data/foodmix_sa1b/annotations" "アノテーションディレクトリ" && \
   check_file "data/foodmix_sa1b/train.txt" "学習データリスト" && \
   check_file "data/foodmix_sa1b/val.txt" "検証データリスト"; then
    
    TRAIN_COUNT=$(wc -l < data/foodmix_sa1b/train.txt)
    VAL_COUNT=$(wc -l < data/foodmix_sa1b/val.txt)
    echo "   学習データ: $TRAIN_COUNT ファイル"
    echo "   検証データ: $VAL_COUNT ファイル"
    DATA_OK=1
fi
echo ""

echo -e "${BLUE}8. Git管理状況確認${NC}"
echo "----------------------------------------"
echo "未追加のLargeモデル関連ファイル:"
git status --porcelain 2>/dev/null | grep -E "(l_foodmix|large|Large)" | head -10

echo ""
echo "サブモジュール内の未追加ファイル:"
(cd external/sam2 && git status --porcelain 2>/dev/null | grep "l_foodmix" | head -5)
echo ""

echo -e "${BLUE}9. GPU環境確認${NC}"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU情報:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read gpu_info; do
        echo "   $gpu_info MB"
    done
    
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -ge 20000 ]; then
        echo -e "   ${GREEN}メモリ充分: Largeモデル学習に適している${NC}"
    elif [ "$GPU_MEM" -ge 16000 ]; then
        echo -e "   ${YELLOW}メモリ注意: メモリ最適化設定の使用を推奨${NC}"
    else
        echo -e "   ${RED}メモリ不足: Largeモデル学習には困難${NC}"
    fi
else
    echo -e "${YELLOW}nvidia-smi が見つかりません (CPUのみ環境)${NC}"
fi
echo ""

echo "==============================================="
echo -e "${BLUE}総合結果${NC}"
echo "==============================================="

TOTAL_SCORE=0
MAX_SCORE=6

if [ $CHECKPOINT_OK -eq 1 ]; then ((TOTAL_SCORE++)); fi
if [ $MODEL_CONFIG_OK -eq 1 ]; then ((TOTAL_SCORE++)); fi
if [ $MAIN_CONFIG_COUNT -ge 2 ]; then ((TOTAL_SCORE++)); fi
if [ $DETAIL_CONFIG_COUNT -ge 3 ]; then ((TOTAL_SCORE++)); fi
if [ $SCRIPT_OK -eq 1 ]; then ((TOTAL_SCORE++)); fi
if [ $DATA_OK -eq 1 ]; then ((TOTAL_SCORE++)); fi

echo "設定完了度: $TOTAL_SCORE/$MAX_SCORE"

if [ $TOTAL_SCORE -eq $MAX_SCORE ]; then
    echo -e "${GREEN}✓ SAM2.1 Largeモデル環境は完全にセットアップされています！${NC}"
    echo ""
    echo "次のコマンドで学習を開始できます:"
    echo "  bash scripts/20_train_food_sam2_large.sh"
elif [ $TOTAL_SCORE -ge 4 ]; then
    echo -e "${YELLOW}⚠ SAM2.1 Largeモデル環境はほぼ準備できています${NC}"
    echo "不足している項目を確認して完了してください"
else
    echo -e "${RED}✗ SAM2.1 Largeモデル環境の設定が不完全です${NC}"
    echo "上記のエラー項目を修正してください"
fi

echo ""
echo "詳細情報:"
echo "  - 最適化ガイド: docs/large_model_optimization_guide.md"
echo "  - 設定対応表: docs/large_model_config_mapping.md"
echo "  - ファイル配置: docs/large_model_file_placement.md"