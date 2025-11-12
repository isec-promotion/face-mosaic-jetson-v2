#!/usr/bin/env python3
"""
シンプルなRTSP→YouTube Live配信プログラム
顔検出なし、最小構成でテスト用

使い方:
    python3 simple_rtsp_youtube.py "<RTSP_URL>" "<YOUTUBE_STREAM_KEY>"
"""
import argparse
import logging
import signal
import sys

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

LOG = logging.getLogger("simple-rtsp-youtube")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_BITRATE = 6_000_000  # 6 Mbps (1080p推奨)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTSP映像をそのままYouTube Liveへ配信（テスト用）"
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


def make_element(factory: str, name: str, **properties) -> Gst.Element:
    element = Gst.ElementFactory.make(factory, name)
    if not element:
        raise RuntimeError(f"GStreamer要素の作成に失敗: '{factory}' ({name})")
    for key, value in properties.items():
        element.set_property(key, value)
    return element


def on_pad_added(decodebin, pad, target_element):
    """uridecodebin のパッドが追加されたときの処理"""
    caps = pad.get_current_caps()
    if not caps:
        return
    
    caps_struct = caps.get_structure(0)
    caps_name = caps_struct.get_name()
    
    LOG.debug(f"パッド追加: {caps_name}")
    
    if caps_name.startswith("video"):
        sink_pad = target_element.get_static_pad("sink")
        if not sink_pad.is_linked():
            ret = pad.link(sink_pad)
            if ret == Gst.PadLinkReturn.OK:
                LOG.info("ビデオパッドをリンクしました")
            else:
                LOG.error(f"パッドのリンクに失敗: {ret}")


def bus_call(bus, message, loop, pipeline):
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        LOG.info("ストリーム終了")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        LOG.error(f"エラー: {err} ({debug})")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        LOG.warning(f"警告: {warn} ({debug})")
    elif msg_type == Gst.MessageType.STATE_CHANGED:
        if message.src == pipeline:
            old, new, pending = message.parse_state_changed()
            LOG.info(f"パイプライン状態変化: {old.value_nick} -> {new.value_nick}")
    return True


def build_pipeline(args: argparse.Namespace) -> Gst.Pipeline:
    """
    シンプルなパイプライン構成:
    uridecodebin -> nvvideoconvert -> capsfilter -> nvv4l2h264enc -> h264parse -> flvmux -> rtmpsink
    """
    pipeline = Gst.Pipeline.new("simple-rtsp-youtube-pipeline")
    if not pipeline:
        raise RuntimeError("パイプラインの作成に失敗")

    # RTSP ソース
    source = make_element("uridecodebin", "rtsp-source", uri=args.rtsp_url)
    
    # ビデオ変換（NVMM対応）
    nvvidconv = make_element("nvvideoconvert", "nv-video-converter")
    
    # キュー（バッファリング用）
    queue1 = make_element("queue", "queue1")
    
    # エンコーダー用のフォーマット指定
    caps_str = "video/x-raw(memory:NVMM), format=(string)NV12"
    capsfilter = make_element(
        "capsfilter",
        "caps-filter",
        caps=Gst.Caps.from_string(caps_str)
    )
    
    # キュー2
    queue2 = make_element("queue", "queue2")
    
    # H.264 エンコーダー（Jetson HWエンコーダー）
    # iframeinterval: YouTubeは2秒以下を推奨（30fps * 2 = 60フレーム = 2秒）
    encoder = make_element(
        "nvv4l2h264enc",
        "hw-encoder",
        bitrate=args.bitrate,
        insert_sps_pps=True,
        iframeinterval=30,  # 1秒間隔（より安全な設定）
        profile=0,  # Baseline Profile
    )
    
    # H.264 パーサー
    h264parser = make_element("h264parse", "h264-parser")
    
    # キュー3（エンコーダー後）
    queue3 = make_element("queue", "queue3")
    
    # ダミー音声ソース（YouTubeは音声ストリームを要求する）
    audiotestsrc = make_element("audiotestsrc", "audio-source", wave=4)  # wave=4は無音
    audioconvert = make_element("audioconvert", "audio-convert")
    audioresample = make_element("audioresample", "audio-resample")
    voaacenc = make_element("voaacenc", "audio-encoder", bitrate=128000)
    aacparse = make_element("aacparse", "aac-parser")
    audio_queue = make_element("queue", "audio-queue")
    
    # FLV muxer
    flvmux = make_element("flvmux", "flv-muxer", streamable=True)
    
    # RTMP sink（YouTube）
    sink = make_element(
        "rtmpsink",
        "youtube-sink",
        location=args.youtube_url,
    )
    
    # パイプラインに要素を追加
    for elem in (source, nvvidconv, queue1, capsfilter, queue2, encoder, h264parser, queue3,
                 audiotestsrc, audioconvert, audioresample, voaacenc, aacparse, audio_queue,
                 flvmux, sink):
        pipeline.add(elem)
    
    # 要素をリンク（sourceは動的にリンク）
    if not nvvidconv.link(queue1):
        raise RuntimeError("nvvideoconvert -> queue1 のリンクに失敗")
    if not queue1.link(capsfilter):
        raise RuntimeError("queue1 -> capsfilter のリンクに失敗")
    if not capsfilter.link(queue2):
        raise RuntimeError("capsfilter -> queue2 のリンクに失敗")
    if not queue2.link(encoder):
        raise RuntimeError("queue2 -> encoder のリンクに失敗")
    if not encoder.link(h264parser):
        raise RuntimeError("encoder -> h264parse のリンクに失敗")
    if not h264parser.link(queue3):
        raise RuntimeError("h264parse -> queue3 のリンクに失敗")
    # ビデオストリームをflvmuxに接続
    video_pad = flvmux.get_request_pad("video")
    queue3_src = queue3.get_static_pad("src")
    if queue3_src.link(video_pad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("queue3 -> flvmux (video) のリンクに失敗")
    
    # 音声ストリームをリンク
    if not audiotestsrc.link(audioconvert):
        raise RuntimeError("audiotestsrc -> audioconvert のリンクに失敗")
    if not audioconvert.link(audioresample):
        raise RuntimeError("audioconvert -> audioresample のリンクに失敗")
    if not audioresample.link(voaacenc):
        raise RuntimeError("audioresample -> voaacenc のリンクに失敗")
    if not voaacenc.link(aacparse):
        raise RuntimeError("voaacenc -> aacparse のリンクに失敗")
    if not aacparse.link(audio_queue):
        raise RuntimeError("aacparse -> audio_queue のリンクに失敗")
    
    # 音声ストリームをflvmuxに接続
    audio_pad = flvmux.get_request_pad("audio")
    audio_queue_src = audio_queue.get_static_pad("src")
    if audio_queue_src.link(audio_pad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("audio_queue -> flvmux (audio) のリンクに失敗")
    
    # flvmuxからrtmpsinkへ
    if not flvmux.link(sink):
        raise RuntimeError("flvmux -> rtmpsink のリンクに失敗")
    
    # uridecodebin の pad-added シグナルを接続
    source.connect("pad-added", on_pad_added, nvvidconv)
    
    LOG.info("パイプライン構築完了")
    return pipeline


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    
    Gst.init(None)
    
    pipeline = build_pipeline(args)
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop, pipeline)
    
    def handle_sigint(signum, frame):
        LOG.info(f"シグナル {signum} を受信、停止中...")
        loop.quit()
    
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)
    
    try:
        LOG.info(f"パイプライン開始 -> {args.youtube_url}")
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            LOG.error("パイプラインの開始に失敗しました")
            return 1
        
        loop.run()
    except KeyboardInterrupt:
        LOG.info("キーボード割り込みを受信")
    finally:
        LOG.info("パイプライン停止中")
        pipeline.set_state(Gst.State.NULL)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
