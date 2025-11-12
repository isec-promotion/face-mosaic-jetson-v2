#!/usr/bin/env python3
"""
RTSP→顔モザイク→YouTube Live配信プログラム
YOLOv8n-face を使用した顔検出版

使い方:
    python3 deepstream_youtube-v2.py "<RTSP_URL>" "<YOUTUBE_STREAM_KEY>"
"""
import argparse
import logging
import signal
import sys
import cv2
import numpy as np
import threading
from queue import Queue, Full
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

try:
    from ultralytics import YOLO
except ImportError:
    print("エラー: ultralyticsパッケージがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("  pip install ultralytics")
    sys.exit(1)

LOG = logging.getLogger("face-mosaic-streamer")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_BITRATE = 6_000_000  # 6 Mbps (1080p推奨)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTSP映像から顔にモザイクをかけてYouTube Liveへ配信"
    )
    parser.add_argument(
        "rtsp_url",
        help="RTSP URL (例: rtsp://user:pass@ip:554/Streaming/Channels/101)",
    )
    parser.add_argument(
        "stream_key",
        help="YouTube Live stream key",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"出力幅 (デフォルト: {DEFAULT_WIDTH}, 720pの場合は1280を指定)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"出力高さ (デフォルト: {DEFAULT_HEIGHT}, 720pの場合は720を指定)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"出力FPS (デフォルト: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=DEFAULT_BITRATE,
        help=f"H.264エンコーダービットレート (デフォルト: {DEFAULT_BITRATE}, 720pの場合は2500000を推奨)",
    )
    parser.add_argument(
        "--youtube-ingest",
        default="rtmp://a.rtmp.youtube.com/live2",
        help="YouTube RTMPエンドポイント (デフォルト: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="顔検出の信頼度閾値 (デフォルト: 0.5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル",
    )
    args = parser.parse_args()
    args.youtube_url = f"{args.youtube_ingest}/{args.stream_key}"
    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def apply_mosaic(image, x, y, w, h, ratio=0.05):
    """指定された領域にモザイク処理を適用"""
    # 境界チェック
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return image
    
    # モザイク処理する領域を抽出
    face_img = image[y:y+h, x:x+w]
    
    if face_img.size == 0:
        return image
    
    # 縮小してから拡大することでモザイク効果を作成
    small = cv2.resize(face_img, None, fx=ratio, fy=ratio, 
                      interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), 
                       interpolation=cv2.INTER_NEAREST)
    
    # 元の画像にモザイクを適用
    image[y:y+h, x:x+w] = mosaic
    
    return image


class FaceDetector:
    """YOLOv8n-faceを使った顔検出"""
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        
        # YOLOv8n-faceモデルのパス
        model_path = Path("models/yolov8n-face.pt")
        
        if not model_path.exists():
            LOG.error(f"YOLOv8n-faceモデルが見つかりません: {model_path}")
            LOG.error("以下のコマンドでモデルをダウンロードしてください:")
            LOG.error("  wget https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt -O models/yolov8n-face.pt")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        LOG.info(f"YOLOv8n-faceモデルを読み込み中: {model_path}")
        
        # YOLOv8モデルを読み込み
        self.model = YOLO(str(model_path))
        
        # GPU利用可能かチェック
        import torch
        if torch.cuda.is_available():
            LOG.info("CUDA利用可能 - GPUで顔検出を実行します")
            self.model.to('cuda')
        else:
            LOG.info("CUDA利用不可 - CPUで顔検出を実行します")
        
        LOG.info("YOLOv8n-face読み込み完了")
    
    def detect(self, frame):
        """フレーム内の顔を検出"""
        # YOLOv8で推論（クラス0のみ=顔）
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # バウンディングボックスの座標を取得
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # x, y, w, h形式に変換
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                
                # 境界チェック
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                if w > 0 and h > 0:
                    faces.append((x, y, w, h))
        
        return faces


def process_frames(frame_queue, processed_queue, detector, stop_event):
    """フレーム処理スレッド"""
    LOG.info("フレーム処理スレッドを開始")
    frame_count = 0
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            
            # 顔検出
            faces = detector.detect(frame)
            
            # モザイク処理
            for (x, y, w, h) in faces:
                frame = apply_mosaic(frame, x, y, w, h, ratio=0.05)
            
            # 処理済みフレームをキューに追加
            try:
                processed_queue.put(frame, timeout=0.1)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    LOG.info(f"処理済み: {frame_count}フレーム, 検出顔数: {len(faces)}")
            except Full:
                pass  # キューがいっぱいの場合はスキップ
                
        except:
            continue
    
    LOG.info("フレーム処理スレッドを終了")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    
    LOG.info("=" * 70)
    LOG.info("RTSP→顔モザイク→YouTube Live配信")
    LOG.info("=" * 70)
    LOG.info(f"入力: {args.rtsp_url}")
    LOG.info(f"出力: rtmp://a.rtmp.youtube.com/live2/****")
    LOG.info(f"解像度: {args.width}x{args.height} @ {args.fps}fps")
    LOG.info(f"ビットレート: {args.bitrate}")
    LOG.info("=" * 70)
    
    # 顔検出器を初期化
    detector = FaceDetector(confidence_threshold=args.confidence)
    
    # フレームキュー
    frame_queue = Queue(maxsize=5)
    processed_queue = Queue(maxsize=5)
    stop_event = threading.Event()
    
    # フレーム処理スレッドを開始
    process_thread = threading.Thread(
        target=process_frames,
        args=(frame_queue, processed_queue, detector, stop_event),
        daemon=True
    )
    process_thread.start()
    
    # OpenCVでRTSP取得
    LOG.info("RTSP接続中...")
    cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        LOG.error("RTSP接続に失敗しました")
        return 1
    
    LOG.info("RTSP接続成功")
    
    # GStreamer パイプラインでYouTubeへ配信
    # appsrc → nvvideoconvert → encoder → flvmux → rtmpsink
    pipeline_str = (
        f"appsrc name=src format=3 is-live=true block=true "
        f"caps=video/x-raw,format=BGR,width={args.width},height={args.height},framerate={args.fps}/1 ! "
        f"videoconvert ! "
        f"nvvideoconvert ! "
        f"video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvv4l2h264enc bitrate={args.bitrate} insert-sps-pps=true iframeinterval=30 profile=0 ! "
        f"h264parse ! "
        f"queue ! "
        f"flvmux name=mux streamable=true ! "
        f"rtmpsink location={args.youtube_url} sync=false "
        f"audiotestsrc wave=4 ! audioconvert ! audioresample ! voaacenc bitrate=128000 ! "
        f"aacparse ! queue ! mux."
    )
    
    LOG.info("GStreamerパイプラインを初期化中...")
    Gst.init(None)
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("src")
    
    pipeline.set_state(Gst.State.PLAYING)
    LOG.info("YouTube配信を開始しました")
    LOG.info("処理中... (Ctrl+Cで終了)")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                LOG.warning("フレーム取得失敗、再接続を試みます...")
                cap.release()
                cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            # リサイズ
            frame = cv2.resize(frame, (args.width, args.height))
            
            # フレーム処理キューに追加
            try:
                frame_queue.put(frame.copy(), timeout=0.01)
            except Full:
                pass
            
            # 処理済みフレームを取得
            try:
                processed_frame = processed_queue.get(timeout=0.01)
            except:
                processed_frame = frame  # 処理済みがない場合は元のフレーム
            
            # GStreamerへ送信
            data = processed_frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.pts = frame_count * (Gst.SECOND // args.fps)
            buf.duration = Gst.SECOND // args.fps
            
            ret = appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                LOG.error("フレーム送信に失敗しました")
                break
            
            frame_count += 1
            
    except KeyboardInterrupt:
        LOG.info("キーボード割り込みを受信しました")
    finally:
        LOG.info("クリーンアップ中...")
        stop_event.set()
        process_thread.join(timeout=2)
        cap.release()
        appsrc.emit("end-of-stream")
        pipeline.set_state(Gst.State.NULL)
        LOG.info("終了しました")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
