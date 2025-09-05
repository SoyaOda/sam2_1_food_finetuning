#!/bin/bash
# 00_setup_env.sh - SAM2.1環境セットアップスクリプト
set -eux

echo "SAM2.1 Food Finetuning環境セットアップを開始します..."

# プロジェクトルートを取得
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Python環境確認
echo "Python環境を確認中..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "エラー: Python $required_version 以上が必要です。現在のバージョン: $python_version"
    echo "conda create -n foodsammix python=3.10 -y && conda activate foodsammix を実行してください"
    exit 1
fi

# pipアップグレード
echo "pipをアップグレード中..."
pip install --upgrade pip

# PyTorchインストール確認
echo "PyTorchのインストールを確認中..."
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "PyTorchが既にインストールされています"
    torch_version=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorchバージョン: $torch_version"
    
    # バージョン確認（2.5.1以上か）
    if python3 -c "import torch; import sys; sys.exit(0 if tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 5) else 1)"; then
        echo "PyTorch 2.5.1以上が確認されました"
    else
        echo "警告: PyTorch 2.5.1以上が推奨されています"
        echo "アップグレードする場合は以下を実行してください:"
        echo "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124"
    fi
else
    echo "PyTorchがインストールされていません"
    echo "GPU環境に合わせてインストールしてください:"
    echo "CUDA 12.4: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124"
    echo "CUDA 12.1: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121"
    echo "CPU only: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu"
    exit 1
fi

# 補助ライブラリのインストール
echo "補助ライブラリをインストール中..."
pip install opencv-python numpy pycocotools scikit-image pillow tqdm hydra-core omegaconf matplotlib

# SAM2.1本体のセットアップ
echo "SAM2.1をセットアップ中..."
mkdir -p external
cd external

if [ ! -d "sam2" ]; then
    echo "SAM2リポジトリをクローン中..."
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    
    echo "SAM2を開発モードでインストール中..."
    pip install -e ".[dev]"
    
    # チェックポイントのダウンロード
    echo "SAM2.1チェックポイントをダウンロード中..."
    cd checkpoints
    if [ -f "download_ckpts.sh" ]; then
        chmod +x download_ckpts.sh
        ./download_ckpts.sh
    else
        echo "警告: download_ckpts.shが見つかりません"
        echo "手動でチェックポイントをダウンロードしてください"
    fi
    cd ..
else
    echo "SAM2ディレクトリが既に存在します"
    cd sam2
    
    # 既存のインストールを確認
    if python3 -c "import sam2" 2>/dev/null; then
        echo "SAM2が既にインストールされています"
    else
        echo "SAM2を開発モードでインストール中..."
        pip install -e ".[dev]"
    fi
fi

cd "$PROJECT_ROOT"

# インストール確認
echo ""
echo "=== インストール確認 ==="
echo -n "Python: "
python3 --version
echo -n "PyTorch: "
python3 -c "import torch; print(torch.__version__)"
echo -n "OpenCV: "
python3 -c "import cv2; print(cv2.__version__)"
echo -n "NumPy: "
python3 -c "import numpy; print(numpy.__version__)"
echo -n "Pillow: "
python3 -c "import PIL; print(PIL.__version__)"

# SAM2インポート確認
echo -n "SAM2: "
if python3 -c "import sam2" 2>/dev/null; then
    echo "OK"
else
    echo "エラー: SAM2のインポートに失敗しました"
    exit 1
fi

echo ""
echo "環境セットアップが完了しました！"
echo "次のステップ:"
echo "1. bash scripts/01_download_foodseg103.sh - FoodSeg103データセットのダウンロード"
echo "2. bash scripts/02_download_uecfoodpix.sh - UEC-FoodPixデータセットのダウンロード"