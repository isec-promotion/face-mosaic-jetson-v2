# Jetson Orin NX Super - DeepStream顔モザイクYouTubeリレー

DeepStreamとGStreamerを使ってRTSP監視映像を取り込み、GPU上で顔を検出して塗りつぶしマスクを適用し、ハードウェアエンコードした映像をRTMPS経由でYouTube Liveへ中継する仕組みです。デコードから推論、マスク、エンコードまでNVMMメモリ内で完結するため、Jetson Orin NX Superの性能を無駄なく活用できます。

## アーキテクチャ

- **入力**: `uridecodebin`がRTSPを取得し、`nvv4l2decoder`でHWデコード。
- **推論**: `nvinfer`がNVIDIA提供のResNet18ベース顔検出モデル（TensorRT FP16）を実行。
- **マスキング**: `nvosd`に仕込んだ`pyds`プローブで顔領域をGPU上の塗りつぶし矩形に置換（CPUコピーなし）。
- **出力**: `nvv4l2h264enc`でHWエンコード -> `flvmux` -> `rtmpsink`でYouTubeへ送信。

```
RTSP -> uridecodebin -> nvstreammux -> nvinfer -> nvosd(+probe) -> nvvideoconvert
      -> capsfilter -> nvv4l2h264enc -> h264parse -> flvmux -> rtmpsink (YouTube)
```

## リポジトリ構成

| ファイル | 役割 |
| ---- | ---- |
| `deepstream_youtube.py` | DeepStreamパイプラインを組み立てて制御するPythonランチャー。 |
| `config_infer_primary_facedetect.txt` | NGCのResNet18顔検出モデルを参照する`nvinfer`設定。 |

> ダウンロードした`tlt_facedetect_pruned_v1.0/`フォルダを同じディレクトリに配置してください。

## 事前準備

- Jetson Orin NX Super + JetPack 6.x（CUDA/TensorRT/Multimedia同梱）。
- DeepStream SDK 6.4以降（Pythonバインディング付き、`sudo apt install deepstream-6.4`後に付属仮想環境で`pip install pyds`）。
- Git、Python 3.8+（JetPack標準）、RTSPカメラとYouTube Liveストリームキー。
- （任意）`screen`や`systemd`など常駐実行環境。

## モデル準備

1. Jetson上でNGC CLI (ARM64) を導入:
   ```bash
   wget https://ngc.nvidia.com/downloads/ngccli_arm64.zip
   unzip ngccli_arm64.zip && chmod +x ngc-cli/ngc
   echo 'export PATH=$PATH:$PWD/ngc-cli' >> ~/.bashrc
   source ~/.bashrc
   ngc config set  # APIキーを入力
   ```
2. プルーニング済み顔検出モデルを取得:
   ```bash
   ngc registry model download-version nvidia/tlt_facedetect:pruned_v1.0 -d .
   ```
   `tlt_facedetect_pruned_v1.0/`配下に`.etlt`やキャリブレーション、ラベルが展開されます。
3. `config_infer_primary_facedetect.txt`内の`model-engine-file`で指定した場所に初回実行時`.engine`が自動生成されます。出力先が書き込み可能か確認してください。

## 設定のポイント

- `custom-lib-path`はDeepStream同梱の`/opt/nvidia/deepstream/deepstream/lib/libnvdsparsebbox_facedetect.so`を指します。
- `network-mode=2`でFP16推論を有効化し、Orin NX Super向けに精度と速度のバランスを確保します。
- 本スクリプトは黒塗り矩形で顔を隠します。モザイク化が必須の場合はVPIやPyCUDAを使った独自CUDAカーネルをpad probe内で実装してください。

## 実行手順

```bash
python3 deepstream_youtube.py \
    "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" \
    "YOUR-YOUTUBE-STREAM-KEY"
```

主なオプション:

- `--infer-config`: 別の`nvinfer`設定ファイルを使用。
- `--width/--height/--fps`: 中継解像度やフレームレートを上書き（既定1280x720@30）。
- `--bitrate`: HWエンコーダのビットレート（既定2.5 Mbps CBR）。
- `--youtube-ingest`: YouTubeのRTMPSエンドポイントを変更（既定`rtmps://a.rtmps.youtube.com:443/live2`）。

### 想定されるログ

- 入力/出力設定をまとめたバナー。
- GStreamerバスメッセージ（`INFO`/`WARNING`/`ERROR`）。
- 初回実行時のTensorRTエンジン自動ビルド（約1分）。

`Ctrl+C`で終了できます。SIGINT/SIGTERMを補足し、パイプラインを安全に停止します。

## 運用ヒント

- 可能な限り有線LANを使用し、Wi-FiによるRTMPアンダーランを回避。
- `tegrastats`でエンコーダ負荷を監視（720p30 FP16でGPU 35%未満が目安）。
- 複数カメラ対応は`nvstreammux`の`request_pad`を追加し、`gie-unique-id`を重複させないよう拡張。
- YouTube側で拒否される場合はエンドポイント、ストリームキー、システム時刻（`sudo timedatectl set-ntp true`）を再確認。

## トラブルシュート

- **顔が検出されない**: スクリプト横にNGCモデルディレクトリがあるか、設定パスが正しいか、`libnvdsparsebbox_facedetect.so`が存在するか確認。
- **`nvinfer`のエンジン生成失敗**: 既存`.engine`を削除して再実行。JetPack/DeepStreamのバージョン整合も要確認。
- **音声が必要**: `queue ! audioconvert ! voaacenc ! aacparse ! flvmux.audio`を追加し、音声パイプラインを構築する。
- **黒塗りではなくモザイク**: pad probe内の処理をVPIの`vpiSubmitRescale`や独自CUDAカーネルに置き換え、縮小→拡大でピクセル化を行う。

これらの手順で、Jetsonネイティブかつゼロコピーなプライバシー保護リレーを長時間安定運用できます。
