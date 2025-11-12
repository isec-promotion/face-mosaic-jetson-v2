# Jetson Orin NX Super - DeepStream 顔モザイク YouTube リレー

DeepStream と GStreamer を使って RTSP 監視映像を取り込み、GPU 上で顔を検出して塗りつぶしマスクを適用し、ハードウェアエンコードした映像を RTMPS 経由で YouTube Live へ中継する仕組みです。デコードから推論、マスク、エンコードまで NVMM メモリ内で完結するため、Jetson Orin NX Super の性能を無駄なく活用できます。

## アーキテクチャ

- **入力**: `uridecodebin`が RTSP を取得し、`nvv4l2decoder`で HW デコード。
- **推論**: `nvinfer`が NVIDIA 提供の ResNet18 ベース顔検出モデル（TensorRT FP16）を実行。
- **マスキング**: `nvosd`に仕込んだ`pyds`プローブで顔領域を GPU 上の塗りつぶし矩形に置換（CPU コピーなし）。
- **出力**: `nvv4l2h264enc`で HW エンコード -> `flvmux` -> `rtmpsink`で YouTube へ送信。

```
RTSP -> uridecodebin -> nvstreammux -> nvinfer -> nvosd(+probe) -> nvvideoconvert
      -> capsfilter -> nvv4l2h264enc -> h264parse -> flvmux -> rtmpsink (YouTube)
```

## リポジトリ構成

| ファイル/ディレクトリ                 | 役割                                                            |
| ------------------------------------- | --------------------------------------------------------------- |
| `deepstream_youtube.py`               | DeepStream パイプラインを組み立てて制御する Python ランチャー。 |
| `config_infer_primary_facedetect.txt` | YOLOv8 顔検出モデルを使用する`nvinfer`設定ファイル。            |
| `models/`                             | モデルファイル格納ディレクトリ（.pt、.onnx、.engine など）。    |
| `models/labels_face.txt`              | 顔検出クラスのラベルファイル。                                  |
| `scripts/convert_yolo_to_onnx.py`     | PyTorch モデルを ONNX 形式に変換するスクリプト。                |
| `.gitignore`                          | モデルファイルを Git 管理から除外する設定。                     |

## 事前準備

- Jetson Orin NX Super + JetPack 6.x（CUDA/TensorRT/Multimedia 同梱）
- DeepStream SDK 7.1 以降（Python バインディング付き）
- Git、Python 3.8+（JetPack 標準）
- RTSP カメラと YouTube Live ストリームキー
- （任意）`screen`や`systemd`など常駐実行環境

### 依存パッケージのインストール

```bash
# DeepStreamのインストール（未インストールの場合）
sudo apt install deepstream-7.1

# Python依存パッケージ
pip3 install ultralytics pyds
```

## モデル準備

### 1. YOLOv8n-face モデルのダウンロード

YOLOv8n-face モデルをダウンロードし、`models/`ディレクトリに配置してください：

```bash
# 推奨: Hugging Faceからダウンロード
wget https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt -O models/yolov8n-face.pt

# または、GitHubリリースからダウンロード
# wget https://github.com/akanametov/yolov8-face/releases/download/v1.0/yolov8n-face.pt -O models/yolov8n-face.pt

# ダウンロード後、ファイルサイズを確認（約6MB程度が目安）
ls -lh models/yolov8n-face.pt
```

> **重要**: ダウンロードが完了したら、ファイルサイズを必ず確認してください。  
> 不完全なダウンロードや破損したファイルは、変換時にエラーが発生します。

> **注**: YOLOv8n-face モデルは顔検出用にファインチューニングされた YOLOv8n モデルです。  
> 上記以外にも、各種モデルハブから入手できます。

### 2. ONNX への変換

PyTorch モデル(.pt)を ONNX 形式に変換します：

```bash
python3 scripts/convert_yolo_to_onnx.py
```

変換が成功すると、`models/yolov8n-face.onnx`が生成されます。

### 3. DeepStream カスタムパーサーライブラリのビルド

DeepStream 7.1 では、カスタムパーサーライブラリを手動でビルドする必要があります：

```bash
# カスタムパーサーディレクトリに移動
cd /opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser/

# ビルド（CUDA_VERはnvcc --versionで確認したバージョンを指定）
sudo CUDA_VER=12.6 make

# libディレクトリにインストール
sudo make install

# インストールを確認
ls -lh /opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_infercustomparser.so
```

### 4. TensorRT エンジンの生成

初回実行時に、ONNX ファイルから自動的に TensorRT エンジン(.engine)が生成されます。
この処理には数分かかる場合がありますが、2 回目以降は既存の engine ファイルが使用されます。

## 設定のポイント

- YOLOv8 カスタムライブラリ（`/opt/nvidia/deepstream/deepstream-7.1/lib/libnvdsinfer_custom_impl_yolo.so`）を使用して顔検出を実行します
- `network-mode=2`で FP16 推論を有効化し、Orin NX Super 向けに精度と速度のバランスを確保します
- 入力解像度は 640x640、検出閾値は 0.30 に設定されています
- 本スクリプトは黒塗り矩形で顔を隠します。モザイク化が必須の場合は VPI や PyCUDA を使った独自 CUDA カーネルを pad probe 内で実装してください

## 実行手順

### テストプログラム（顔検出なし）

まず、シンプルな RTSP→YouTube 配信プログラムで基本的な動作を確認してください：

```bash
python3 simple_rtsp_youtube.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    "YOUR-YOUTUBE-STREAM-KEY"
```

このプログラムは顔検出や DeepStream の nvstreammux を使わず、最小構成で動作します。
まずこちらで配信が成功することを確認してから、本番の顔モザイクプログラムに進んでください。

### 本番プログラム（顔モザイク付き）

```bash
python3 deepstream_youtube.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    "YOUR-YOUTUBE-STREAM-KEY"
```

主なオプション:

- `--infer-config`: 別の`nvinfer`設定ファイルを使用。
- `--width/--height/--fps`: 中継解像度やフレームレートを上書き（既定 1280x720@30）。
- `--bitrate`: HW エンコーダのビットレート（既定 2.5 Mbps CBR）。
- `--youtube-ingest`: YouTube の RTMPS エンドポイントを変更（既定`rtmps://a.rtmps.youtube.com:443/live2`）。

### 想定されるログ

- 入力/出力設定をまとめたバナー。
- GStreamer バスメッセージ（`INFO`/`WARNING`/`ERROR`）。
- 初回実行時の TensorRT エンジン自動ビルド（約 1 分）。

`Ctrl+C`で終了できます。SIGINT/SIGTERM を補足し、パイプラインを安全に停止します。

## 運用ヒント

- 可能な限り有線 LAN を使用し、Wi-Fi による RTMP アンダーランを回避。
- `tegrastats`でエンコーダ負荷を監視（720p30 FP16 で GPU 35%未満が目安）。
- 複数カメラ対応は`nvstreammux`の`request_pad`を追加し、`gie-unique-id`を重複させないよう拡張。
- YouTube 側で拒否される場合はエンドポイント、ストリームキー、システム時刻（`sudo timedatectl set-ntp true`）を再確認。

## トラブルシュート

### モデル関連

- **ONNX 変換時の EOFError（"Ran out of input"）**:

  - モデルファイル（.pt）が破損しているか、ダウンロードが不完全です
  - 解決方法:

    ```bash
    # 既存のモデルファイルを削除
    rm models/yolov8n-face.pt

    # 再度ダウンロード
    wget https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt -O models/yolov8n-face.pt

    # ファイルサイズを確認（約6MB程度が目安）
    ls -lh models/yolov8n-face.pt

    # 再度変換を試行
    python3 scripts/convert_yolo_to_onnx.py
    ```

- **ONNX ファイルが見つからない**: `scripts/convert_yolo_to_onnx.py`を実行して ONNX ファイルを生成してください

- **YOLO カスタムライブラリのエラー**:

  - DeepStream 7.1 が正しくインストールされているか確認してください
  - カスタムライブラリパスを確認してください:
    ```bash
    ls -l /opt/nvidia/deepstream/deepstream-7.1/lib/libnvdsinfer_custom_impl_yolo.so
    ```
  - ファイルが存在しない場合は、DeepStream のバージョンに応じてパスを調整してください

- **顔が検出されない**:
  - YOLOv8n-face モデルが正しく配置されているか確認
  - 検出閾値（`pre-cluster-threshold`）を調整してみてください
  - ONNX ファイルと engine ファイルを削除して再生成してください

### DeepStream 関連

- **`nvinfer`のエンジン生成失敗**:
  - 既存の`.engine`ファイルを削除して再実行
  - JetPack/DeepStream のバージョン整合を確認
  - ディスク空き容量を確認（エンジン生成には一時的に大きな空き容量が必要）

### その他

- **音声が必要**: `queue ! audioconvert ! voaacenc ! aacparse ! flvmux.audio`を追加し、音声パイプラインを構築する
- **黒塗りではなくモザイク**: pad probe 内の処理を VPI の`vpiSubmitRescale`や独自 CUDA カーネルに置き換え、縮小 → 拡大でピクセル化を行う
- **YouTube 接続エラー**: ストリームキーとエンドポイント URL、システム時刻を確認してください

これらの手順で、Jetson ネイティブかつゼロコピーなプライバシー保護リレーを長時間安定運用できます。
