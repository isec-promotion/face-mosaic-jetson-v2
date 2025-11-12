#!/usr/bin/env python3
"""
YOLOv8 PyTorchモデルをONNX形式に変換するスクリプト

使用方法:
    python3 scripts/convert_yolo_to_onnx.py

必要なパッケージ:
    pip3 install ultralytics
"""

import os
import sys
from pathlib import Path

def convert_yolo_to_onnx():
    """YOLOv8モデル(.pt)をONNX形式に変換"""
    
    # パス設定
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / "models"
    pt_file = models_dir / "yolov8n-face.pt"
    onnx_file = models_dir / "yolov8n-face.onnx"
    
    # モデルファイルの存在確認
    if not pt_file.exists():
        print(f"エラー: モデルファイルが見つかりません: {pt_file}")
        print("\nYOLOv8n-faceモデルをダウンロードしてください:")
        print("1. 公式リポジトリからダウンロード")
        print("2. models/yolov8n-face.pt として配置")
        sys.exit(1)
    
    try:
        from ultralytics import YOLO
        print(f"✓ ultralyticsパッケージが見つかりました")
    except ImportError:
        print("エラー: ultralyticsパッケージがインストールされていません")
        print("\n以下のコマンドでインストールしてください:")
        print("  pip3 install ultralytics")
        sys.exit(1)
    
    print(f"\nYOLOv8モデルをロード中: {pt_file}")
    model = YOLO(str(pt_file))
    
    print(f"ONNX形式に変換中...")
    print(f"  入力: {pt_file}")
    print(f"  出力: {onnx_file}")
    
    # ONNX形式にエクスポート
    # imgsz=640: 入力画像サイズ (DeepStreamの設定と一致させる)
    # opset=12: ONNX opset version
    # simplify=True: ONNXモデルを簡略化
    model.export(
        format='onnx',
        imgsz=640,
        opset=12,
        simplify=True
    )
    
    # 出力ファイルの確認
    if onnx_file.exists():
        file_size = onnx_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ 変換完了!")
        print(f"  ファイルサイズ: {file_size:.2f} MB")
        print(f"  保存先: {onnx_file}")
        print("\n次のステップ:")
        print("  python3 deepstream_youtube.py <RTSP_URL> <STREAM_KEY>")
    else:
        print("\nエラー: ONNX変換に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 → ONNX 変換スクリプト")
    print("=" * 60)
    convert_yolo_to_onnx()
