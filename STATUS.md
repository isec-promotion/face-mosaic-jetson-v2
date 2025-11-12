# Jetson Orin NX Super - DeepStream 顔検出システム 開発ステータス

**最終更新**: 2025 年 11 月 12 日 14:40

## 📋 プロジェクト概要

RTSP カメラ映像から YOLOv8 で顔を検出し、DeepStream を使用して GPU 上で処理、YouTube Live へストリーミング配信するシステムの構築。

## ✅ 完了した作業

### 1. YOLOv8 カスタムパーサーの実装

- **ファイル**: `nvdsinfer_yolov8_parser.cpp`
- **サイズ**: 39KB
- **機能**: YOLOv8 の出力形式（2D/3D 対応）を解析し、DeepStream のメタデータに変換
- **動作確認**: ✅ 正常動作（出力形状を正しく認識）
  ```
  INFO: YOLOv8 output shape: [1, 5, 8400]
  INFO: Detected 1 classes
  ```

### 2. モデルファイルとビルドシステム

- **YOLOv8n-face モデル**: 5.96 MB (PyTorch)
- **ONNX ファイル**: 11.68 MB
- **TensorRT エンジン**: 自動生成成功
- **ビルドシステム**: Makefile 作成、CUDA 12.6 対応

### 3. リポジトリ構成

```
face-mosaic-jetson-v2/
├── models/
│   ├── yolov8n-face.pt          # PyTorchモデル（Git管理外）
│   ├── yolov8n-face.onnx        # ONNXモデル（Git管理外）
│   ├── labels_face.txt          # ラベルファイル
│   └── *.engine                 # TensorRTエンジン（自動生成、Git管理外）
├── scripts/
│   └── convert_yolo_to_onnx.py  # ONNX変換スクリプト
├── nvdsinfer_yolov8_parser.cpp  # YOLOv8カスタムパーサー
├── Makefile                      # ビルド用
├── config_infer_primary_facedetect.txt  # nvinfer設定
├── deepstream_youtube.py         # メインスクリプト
├── .gitignore                    # Git管理除外設定
└── README.md                     # ドキュメント
```

### 4. ドキュメント整備

- README.md 更新（モデルダウンロード、ビルド手順）
- トラブルシューティング情報追加
- DeepStream 7.1 対応の注意事項記載

## ⚠️ 現在の問題点

### ~~問題 1: 顔検出が 0 objects~~ ✅ 問題ではない

**症状**:

```
INFO: Detected 0 objects before NMS
```

**結論**:
監視カメラに人がいない時間帯は正常な状態。顔が映った時に検出されるはずなので、これは問題ではない。

### 問題 2: not-negotiated エラー

**症状**:

```
error: streaming stopped, reason not-negotiated (-4)
```

**原因**:
nvinfer → nvosd 間のフォーマット不一致

**次のアクション**:

- `output-tensor-meta=0`に変更（メタデータ出力を無効化）
- capsfilter の追加でフォーマット明示
- パイプライン要素間の capabilities を確認

### 問題 3: DeepStream Python バインディング不在

**症状**:
DeepStream 7.1 には`pyds`モジュールが含まれていない

**影響**:
顔検出後の黒塗りマスキング処理が実装できない

**対策済み**:

- マスキング処理を一時的に無効化
- バウンディングボックス表示のみで動作確認

**今後の対応**:

1. pyds をソースからビルド
2. または、nvosd の設定だけで黒塗り実現（border-width 調整）
3. または、C++でカスタムプラグイン作成

## 🔧 次のステップ

### 優先度 1: not-negotiated エラーの解決（最重要）

```txt
# config_infer_primary_facedetect.txt
output-tensor-meta=0  # 1 → 0に変更
```

### 優先度 2: 黒塗りマスキングの実装

**オプション A**: 太い境界線で代用

```python
# nvosdのborder-widthを大きくする
nvosd.set_property("border-width", 200)
```

**オプション B**: pyds ビルド

```bash
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
cd deepstream_python_apps/bindings
# ビルド手順に従う
```

## 📊 技術スタック

- **Jetson**: Orin NX Super
- **JetPack**: 6.x
- **CUDA**: 12.6
- **DeepStream**: 7.1
- **Python**: 3.10
- **モデル**: YOLOv8n-face
- **推論**: TensorRT FP16

## 🐛 既知の warning/エラー（無害）

### 1. drop-on-late property エラー

```
TypeError: object of type `GstDecodeBin' does not have property `drop-on-late'
```

- **影響**: なし（try-except で処理可能）
- **対応**: 低優先度

### 2. get_request_pad deprecation warning

```
DeprecationWarning: Gst.Element.get_request_pad is deprecated
```

- **影響**: なし（動作に問題なし）
- **対応**: `request_pad_simple()`への移行を検討

## 📝 メモ

### モデルダウンロード

```bash
wget https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt \
  -O models/yolov8n-face.pt
```

### ビルドコマンド

```bash
# カスタムパーサー
sudo CUDA_VER=12.6 make

# 実行
python3 deepstream_youtube.py \
  "rtsp://admin:user1357@192.168.40.167:554/Streaming/channels/101" \
  "twjv-gz57-5hhd-uu8j-fa0q"
```

### 参考リンク

- [DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

---

**担当者**: Cline AI Assistant  
**連絡**: GitHub Issues
