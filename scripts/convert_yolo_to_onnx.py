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
    
    # ファイルサイズのチェック
    file_size = pt_file.stat().st_size
    if file_size < 1024 * 1024:  # 1MB未満の場合は警告
        print(f"警告: モデルファイルサイズが小さすぎます: {file_size / 1024:.2f} KB")
        print("ファイルが正しくダウンロードされていない可能性があります。")
        print("\n以下を確認してください:")
        print("1. ダウンロードが完了しているか")
        print("2. ファイルが破損していないか")
        print("3. 正しいYOLOv8n-faceモデルファイルか")
        sys.exit(1)
    
    print(f"✓ モデルファイルサイズ: {file_size / (1024 * 1024):.2f} MB")
    
    try:
        from ultralytics import YOLO
        print(f"✓ ultralyticsパッケージが見つかりました")
    except ImportError:
        print("エラー: ultralyticsパッケージがインストールされていません")
        print("\n以下のコマンドでインストールしてください:")
        print("  pip3 install ultralytics")
        sys.exit(1)
    
    print(f"\nYOLOv8モデルをロード中: {pt_file}")
    try:
        model = YOLO(str(pt_file))
    except EOFError:
        print("\nエラー: モデルファイルが破損しているか、不完全です。")
        print("\n考えられる原因:")
        print("1. ダウンロードが途中で中断された")
        print("2. ファイル転送中にエラーが発生した")
        print("3. ストレージの容量が不足している")
        print("\n解決方法:")
        print("1. モデルファイルを削除してください:")
        print(f"   rm {pt_file}")
        print("2. 再度ダウンロードしてください")
        print("3. ダウンロード完了後、ファイルサイズを確認してください")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラー: モデルのロードに失敗しました: {e}")
        print("\n以下を確認してください:")
        print("1. 正しいYOLOv8形式のモデルファイルか")
        print("2. PyTorchとultralyticsのバージョンが互換性があるか")
        sys.exit(1)
    
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
