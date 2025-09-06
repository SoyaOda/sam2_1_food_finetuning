#!/usr/bin/env python3
"""
RLEエンコーディングのデバッグスクリプト
"""

import numpy as np
from pycocotools import mask as mask_utils

def to_rle(bin_mask: np.ndarray):
    """
    バイナリマスクをRLE（Run-Length Encoding）形式に変換
    SA-1B形式のため、Fortran orderでエンコード（エラー対応版）
    """
    if bin_mask.dtype != np.uint8:
        bin_mask = bin_mask.astype(np.uint8)
    
    # Fortran order (列優先) でエンコード - SA-1B形式の要件
    encoded = mask_utils.encode(np.asfortranarray(bin_mask[:, :, None]))
    
    print(f"encoded type: {type(encoded)}")
    print(f"encoded content: {encoded}")
    
    # pycocotoolsの結果が辞書かリストかをチェック
    if isinstance(encoded, dict):
        # 通常の場合（辞書）
        if isinstance(encoded.get("counts"), bytes):
            encoded["counts"] = encoded["counts"].decode("ascii")
        return encoded
    elif isinstance(encoded, list) and len(encoded) == 1:
        # リストの場合（一部のpycocotoolsバージョン）
        rle_dict = encoded[0]
        if isinstance(rle_dict, dict) and "counts" in rle_dict:
            if isinstance(rle_dict["counts"], bytes):
                rle_dict["counts"] = rle_dict["counts"].decode("ascii")
            return rle_dict
        else:
            # 手動でRLE辞書を作成
            return {
                "size": [bin_mask.shape[0], bin_mask.shape[1]],
                "counts": str(encoded)  # フォールバック
            }
    else:
        # その他のケース、手動でRLE辞書を作成
        return {
            "size": [bin_mask.shape[0], bin_mask.shape[1]], 
            "counts": str(encoded)  # フォールバック
        }

# テスト用のバイナリマスクを作成
test_mask = np.zeros((100, 100), dtype=np.uint8)
test_mask[20:50, 30:70] = 1

print("テストマスクの形状:", test_mask.shape)
print("テストマスクの型:", test_mask.dtype)
print("非ゼロ要素数:", np.sum(test_mask))

# RLE変換をテスト
try:
    rle = to_rle(test_mask)
    print("RLE変換成功")
    print("RLE type:", type(rle))
    print("RLE keys:", rle.keys() if hasattr(rle, 'keys') else 'N/A')
    
    # bbox計算をテスト
    bbox_result = mask_utils.toBbox(rle)
    print("bbox計算成功")
    print("bbox type:", type(bbox_result))
    print("bbox content:", bbox_result)
    print("bbox shape:", bbox_result.shape if hasattr(bbox_result, 'shape') else 'N/A')
    
    # bbox変換をテスト
    if hasattr(bbox_result, 'tolist'):
        bbox = bbox_result.tolist()
    elif isinstance(bbox_result, (list, tuple)):
        bbox = list(bbox_result)
    else:
        bbox = [float(bbox_result[i]) for i in range(4)]
    
    print("bbox変換成功:", bbox)
    
except Exception as e:
    print(f"エラー発生: {e}")
    import traceback
    traceback.print_exc()