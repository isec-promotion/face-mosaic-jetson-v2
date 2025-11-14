# Face Mosaic for NVIDIA Jetson with YOLO11

NVIDIA Jetson 上で DeepStream と YOLO11 を使用して、RTSP カメラ映像から顔を検出し、プライバシー保護のために顔領域を黒く塗りつぶすプログラムです。

## 概要

このプロジェクトは、NVIDIA Jetson デバイスで YOLO11 顔検出モデルを使用し、リアルタイムで顔領域を検出してマスク処理を行います。DeepStream SDK と TensorRT を活用することで、効率的な GPU 処理を実現しています。

## 機能

- **RTSP 映像入力**: IP カメラなどの RTSP 映像ソースからの入力に対応
- **GPU 高速処理**: NVMM（NVIDIA Memory Management）を使用したゼロコピー処理
- **YOLO11 顔検出**: 最新の YOLO11 モデルによる高精度な顔検出
- **2 つの動作モード**:
  - ローカル表示モード: 顔検出結果をローカル画面に表示
  - YouTube 配信モード: 顔検出処理後の映像を YouTube Live に配信

## リポジトリ構成

| ファイル/ディレクトリ                  | 役割                                                       |
| -------------------------------------- | ---------------------------------------------------------- |
| `simple_rtsp_local_blackout.py`        | RTSP カメラの映像をローカル画面に表示（顔を黒塗り）        |
| `simple_rtsp_youtube.py`               | RTSP カメラの映像を YouTube Live に配信（顔検出なし）      |
| `config_infer_primary_face_yolo11.txt` | YOLO11 顔検出モデルの nvinfer 設定ファイル                 |
| `models/`                              | モデルファイル格納ディレクトリ（.pt、.onnx、.engine など） |
| `models/labels_face.txt`               | 顔検出クラスのラベルファイル                               |

## システム要件

- **ハードウェア**: NVIDIA Jetson Orin NX / Orin Nano / Xavier NX / Nano（JetPack 4.6.4 以降）
- **ソフトウェア**:
  - JetPack 6.1（推奨）または JetPack 5.1.3 / 4.6.4
  - DeepStream SDK 7.1（JetPack 6.1）/ 6.3（JetPack 5.1.3）/ 6.0.1（JetPack 4.6.4）
  - Python 3.8 以降
  - CUDA、TensorRT（JetPack に含まれる）

## セットアップ

### 1. JetPack と DeepStream SDK のインストール

JetPack のバージョンに応じて、適切な DeepStream SDK をインストールしてください。

```bash
# JetPack 6.1の場合
sudo apt install deepstream-7.1

# JetPack 5.1.3の場合
sudo apt install deepstream-6.3

# JetPack 4.6.4の場合
sudo apt install deepstream-6.0.1
```

詳細は[NVIDIA DeepStream Getting Started](https://developer.nvidia.com/deepstream-getting-started)を参照してください。

### 2. Ultralytics YOLO のインストール

以下のコマンドで Ultralytics と YOLO11 をインストールします:

```bash
cd ~
pip install -U pip
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e ".[export]" onnxslim
```

### 3. DeepStream-YOLO のセットアップ

YOLO11 モデルを DeepStream で使用するために、DeepStream-Yolo リポジトリを使用します:

```bash
cd ~
git clone https://github.com/marcoslucianops/DeepStream-Yolo
```

### 4. YOLO11 モデルの準備

#### 4.1 モデルのダウンロード

YOLO11 の顔検出モデルをダウンロードします:

```bash
cd ~/ultralytics
# YOLO11の標準モデルをダウンロード（例：yolo11s.pt）
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

顔検出用にファインチューニングされたモデルを使用する場合は、該当するモデルファイルをダウンロードしてください。

#### 4.2 ONNX への変換

DeepStream-Yolo のエクスポートスクリプトを Ultralytics ディレクトリにコピーします:

```bash
cp ~/DeepStream-Yolo/utils/export_yolo11.py ~/ultralytics
cd ~/ultralytics
```

モデルを ONNX 形式に変換します:

```bash
# 基本的な変換（FP32）
python3 export_yolo11.py -w yolo11s.pt

# FP16精度で変換（推奨）
python3 export_yolo11.py -w yolo11s.pt --simplify

# 動的バッチサイズを使用する場合（DeepStream 6.1以降）
python3 export_yolo11.py -w yolo11s.pt --dynamic --simplify
```

**変換オプション**:

- `--opset`: ONNX の opset バージョン（デフォルト: 17、DeepStream 5.1 では 12 以下）
- `--simplify`: ONNX モデルを簡略化
- `--dynamic`: 動的バッチサイズを有効化（DeepStream 6.1 以降）
- `-s SIZE` または `--size SIZE`: 入力サイズを指定（デフォルト: 640）

#### 4.3 生成されたファイルをプロジェクトにコピー

```bash
# ONNXファイルとラベルファイルをDeepStream-Yoloにコピー
cp yolo11s.pt.onnx labels.txt ~/DeepStream-Yolo

# DeepStream-Yoloディレクトリに移動
cd ~/DeepStream-Yolo
```

### 5. CUDA 環境変数の設定

JetPack のバージョンに応じて CUDA バージョンを設定します:

```bash
# JetPack 6.1の場合
export CUDA_VER=12.6

# JetPack 5.1.3の場合
export CUDA_VER=11.4

# JetPack 4.6.4の場合
export CUDA_VER=10.2
```

### 6. カスタムライブラリのビルド

```bash
cd ~/DeepStream-Yolo
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo

# システムの動的リンクライブラリキャッシュを更新
sudo ldconfig
```

**重要**: `sudo ldconfig`を実行することで、新しくビルドされたライブラリをシステムが認識できるようになります。

### 7. プロジェクトファイルの準備

生成された ONNX ファイルとビルドされたライブラリをこのプロジェクトのディレクトリにコピーします:

```bash
# このプロジェクトのディレクトリに移動（パスは環境に応じて変更）
cd ~/face-mosaic-jetson-v2

# ONNXファイルをコピー
cp ~/DeepStream-Yolo/yolo11s.pt.onnx models/

# ラベルファイルをコピー（必要に応じて）
cp ~/DeepStream-Yolo/labels.txt models/labels_face.txt

# config_infer_primary_face_yolo11.txtを編集して、正しいパスを設定
```

`config_infer_primary_face_yolo11.txt`内の以下の行を確認・編集してください:

```ini
[property]
onnx-file=models/yolo11s.pt.onnx
labelfile-path=models/labels_face.txt
custom-lib-path=/home/Jetson/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

**重要**: `custom-lib-path`は実際のビルドされたライブラリのパスに変更してください。

### 8. Python 依存パッケージのインストール

```bash
pip3 install pyds gi opencv-python
```

## YOLO11n-face 顔検出モデルの導入（推奨）

YOLO11n-face は顔検出に特化したコミュニティモデルで、YOLO11 の軽量版(nano)をベースに顔検出用にファインチューニングされています。汎用の YOLO11 モデルと比較して、顔検出の精度が高く、処理も高速です。

### YOLO11n-face の特徴

- **高精度**: 顔検出専用にトレーニングされており、小さい顔や角度のある顔も高精度で検出
- **軽量**: nano バージョンのため、Jetson デバイスでもリアルタイム処理が可能
- **1 クラス検出**: 顔のみを検出するため、処理がシンプルで高速

### 1. YOLO11n-face モデルのダウンロード

Hugging Face からコミュニティが公開している YOLO11n-face モデルをダウンロードします:

```bash
# プロジェクトディレクトリに移動
cd ~/face-mosaic-jetson-v2

# モデル格納ディレクトリを作成
mkdir -p models/yolo11n-face

# Hugging Faceからモデルをダウンロード
cd models/yolo11n-face
wget https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/model.pt　-O yolo11n-face.pt
```

**参考**: [YOLO11n-face on Hugging Face](https://huggingface.co/AdamCodd/YOLOv11n-face-detection)

### 2. PT モデルから ONNX への変換

ダウンロードした PyTorch モデル(.pt)を ONNX 形式に変換します:

```bash
# Ultralyticsディレクトリに移動
cd ~/ultralytics

# DeepStream-Yoloのエクスポートスクリプトをコピー（まだの場合）
cp ~/DeepStream-Yolo/utils/export_yolo11.py .

# YOLO11n-faceモデルをONNXに変換
# 出力先を直接指定しているので、後で mv は不要
python3 export_yolo11.py \
    -w ~/face-mosaic-jetson-v2/models/yolo11n-face/yolo11n-face.pt \
    --simplify \
    --dynamic \
    -o ~/face-mosaic-jetson-v2/model　s/yolo11n-face/yolo11n-face.onnx

# これで指定ディレクトリに ONNX が直接生成される
```

**変換オプションの説明**:

- `--simplify`: ONNX モデルを最適化（推奨）
- `--dynamic`: 動的バッチサイズを有効化（DeepStream 6.1 以降）

### 3. TensorRT エンジンファイルの生成

TensorRT エンジンファイルは、プログラムの初回実行時に自動的に生成されます。手動で生成する必要はありませんが、初回実行時には数分かかる場合があります。

生成されるエンジンファイル:

```
models/yolo11n-face/yolo11n-face_b1_fp16.engine
```

### 4. 設定ファイルの確認

`config_infer_primary_face_yolo11n-face.txt`が正しく設定されていることを確認します:

```ini
[property]
onnx-file=./models/yolo11n-face/yolo11n-face.onnx
model-engine-file=./models/yolo11n-face/yolo11n-face_b1_fp16.engine
num-detected-classes=1  # 顔のみ
```

DeepStream-YOLO カスタムライブラリのパスが環境に合っているか確認してください:

```bash
# ライブラリの存在確認
ls -l /opt/nvidia/deepstream/deepstream-7.1/lib/libnvdsinfer_custom_impl_Yolo.so

# パスが異なる場合は設定ファイルを編集
# 例: JetPack 5.1.3の場合
# custom-lib-path=/opt/nvidia/deepstream/deepstream-6.3/lib/libnvdsinfer_custom_impl_Yolo.so
```

### 5. プログラムの実行

YOLO11n-face 専用プログラムを実行します:

```bash
# 基本的な使用方法
python3 simple_rtsp_local_yolo11n-face.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101"

# 解像度とフレームレートを指定
python3 simple_rtsp_local_yolo11n-face.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    --width 1920 --height 1080 --fps 30

# TCP接続を使用
python3 simple_rtsp_local_yolo11n-face.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    --tcp

# デバッグモード（検出情報を詳細表示）
python3 simple_rtsp_local_yolo11n-face.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    --log-level DEBUG
```

**プログラムの終了**: キーボードで `q` を押すか、`Ctrl+C` で終了します。

### トラブルシューティング

#### モデルが見つからない

エラー: `Unable to open model file`

確認事項:

```bash
# ONNXファイルの存在確認
ls -l models/yolo11n-face/yolo11n-face.onnx

# パスが正しいか設定ファイルを確認
cat config_infer_primary_face_yolo11n-face.txt | grep onnx-file
```

#### TensorRT エンジン生成エラー

初回実行時にエンジン生成でエラーが発生した場合:

```bash
# 既存のengineファイルを削除して再生成
rm models/yolo11n-face/*.engine

# プログラムを再実行（エンジンが自動生成される）
python3 simple_rtsp_local_yolo11n-face.py "rtsp://..."
```

#### 顔が検出されない

検出閾値を調整してみてください:

```ini
# config_infer_primary_face_yolo11n-face.txtを編集
[class-attrs-all]
pre-cluster-threshold=0.25  # デフォルトは0.30、より低い値で試す
```

## 使用方法

### ローカル表示モード（simple_rtsp_local_blackout.py）

RTSP カメラの映像をローカル画面に表示しながら、顔を黒く塗りつぶします:

```bash
# 基本的な使用方法
python3 simple_rtsp_local_blackout.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    --infer-config ./config_infer_primary_face_yolo11.txt

# 解像度とフレームレートを指定
python3 simple_rtsp_local_blackout.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    --infer-config ./config_infer_primary_face_yolo11.txt \
    --width 1920 --height 1080 --fps 30

# TCP接続を使用する場合（UDPで接続できない場合）
python3 simple_rtsp_local_blackout.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    --infer-config ./config_infer_primary_face_yolo11.txt \
    --tcp
```

**主なオプション**:

- `--infer-config`: nvinfer の設定ファイルパス（必須）
- `--width/--height/--fps`: 映像の解像度とフレームレート
- `--tcp`: RTSP を TCP 接続で受信（デフォルトは UDP）
- `--latency`: RTSP ソースのレイテンシ（ミリ秒、デフォルト 200）
- `--log-level`: ログレベル（DEBUG/INFO/WARNING/ERROR）

**パイプライン構成**:

```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder
  → nvvideoconvert → capsfilter(NVMM) → nvstreammux
  → nvinfer(YOLO11顔検出) → nvdsosd(黒塗り処理)
  → nvvideoconvert → nvegltransform → nveglglessink
```

### YouTube 配信モード（simple_rtsp_youtube.py）

RTSP カメラの映像を YouTube Live に配信します（顔検出処理なし）:

```bash
python3 simple_rtsp_youtube.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    "YOUR-YOUTUBE-STREAM-KEY"
```

**重要なポイント**:

- YouTube Live への配信には音声ストリームが必須です
- このプログラムはダミー音声（無音）を自動的に追加します
- 配信開始後、YouTube Studio で映像が表示されるまで 10-30 秒かかる場合があります

**パイプライン構成**:

```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder
  → nvvideoconvert → nvv4l2h264enc → h264parse → flvmux
  → rtmpsink (YouTube)
```

## 設定ファイルの説明

### config_infer_primary_face_yolo11.txt

YOLO11 モデルの推論設定を行うファイルです。主要なパラメータ:

- `onnx-file`: ONNX モデルファイルのパス
- `model-engine-file`: TensorRT エンジンファイルのパス（初回実行時に自動生成）
- `custom-lib-path`: YOLO カスタムパーサーライブラリのパス
- `network-mode`: 推論精度（0=FP32, 1=INT8, 2=FP16）
- `num-detected-classes`: 検出クラス数
- `pre-cluster-threshold`: 検出閾値

## トラブルシューティング

### TensorRT エンジンの生成

初回実行時、ONNX ファイルから TensorRT エンジン（.engine）が自動生成されます。この処理には数分かかる場合がありますが、2 回目以降は既存の engine ファイルが使用されます。

エンジン生成でエラーが発生した場合:

```bash
# 既存のengineファイルを削除して再生成
rm models/*.engine
```

### カスタムライブラリが見つからない

エラー: `Unable to open custom lib`

解決方法:

```bash
# ライブラリが正しくビルドされているか確認
ls -l ~/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so

# config_infer_primary_face_yolo11.txt内のcustom-lib-pathを正しいパスに修正
```

### 顔が検出されない

以下を確認してください:

- YOLO11 モデルが顔検出用にトレーニングされているか
- `pre-cluster-threshold`（検出閾値）が適切か（推奨: 0.3-0.5）
- カメラの解像度と距離が適切か

検出閾値を下げてみる:

```ini
[class-attrs-all]
pre-cluster-threshold=0.3  # より低い値で試す
```

### ローカル表示のエラー

エラー: `Could not open display`

解決方法:

```bash
# X11環境を有効化
export DISPLAY=:0

# SSH経由の場合、X11フォワーディングを有効化
ssh -X user@jetson-ip
```

### YouTube 配信のエラー

- ストリームキーが正しいか確認
- インターネット接続が安定しているか確認（有線 LAN 推奨）
- システム時刻が正確か確認: `sudo timedatectl set-ntp true`

## 参考資料

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [NVIDIA DeepStream SDK Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [DeepStream-Yolo GitHub Repository](https://github.com/marcoslucianops/DeepStream-Yolo)
- [YOLO11 on NVIDIA Jetson with DeepStream Guide](https://docs.ultralytics.com/ja/guides/deepstream-nvidia-jetson/)

## ライセンス

このプロジェクトは、関連する各コンポーネントのライセンスに従います。

## 謝辞

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO11 モデル
- [marcoslucianops](https://github.com/marcoslucianops) - DeepStream-Yolo 実装
- NVIDIA - DeepStream SDK and Jetson Platform
