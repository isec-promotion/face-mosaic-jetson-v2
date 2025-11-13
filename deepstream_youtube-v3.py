#!/usr/bin/env python3
import argparse
import logging
import signal
import sys

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

"""
python3 deepstream_youtube-v3.py \
  "rtsp://admin:pass@192.168.xx.xx:554/Streaming/Channels/101" \
  "xxxx-xxxx-xxxx-xxxx" \
  --infer-config ./config_infer_primary_facedet.txt \
  --mosaic-level 6 \
  --bitrate 6000000 --fps 30 --width 1920 --height 1080
"""

LOG = logging.getLogger("rtsp-youtube-face-mosaic-ds")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_BITRATE = 6_000_000

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTSP→YouTube（GPU顔モザイク/DeepStream）")
    p.add_argument("rtsp_url", help="rtsp://user:pass@ip:554/Streaming/Channels/101 等")
    p.add_argument("stream_key", help="YouTube Live stream key")
    p.add_argument("--youtube-ingest", default="rtmp://a.rtmp.youtube.com/live2")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    p.add_argument("--infer-config", default="./config_infer_primary_facedet.txt",
                   help="nvinfer の設定ファイルパス")
    p.add_argument("--mosaic-level", type=int, default=5,
                   help="dsexample のモザイク強度（3〜10目安）")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    args.youtube_url = f"{args.youtube-ingest}/{args.stream_key}"
    return args

def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def make(factory, name, **props):
    e = Gst.ElementFactory.make(factory, name)
    if not e:
        raise RuntimeError(f"要素作成失敗: {factory} ({name})")
    for k, v in props.items():
        e.set_property(k, v)
    return e

def on_pad_added_decodebin(decodebin, pad, data):
    """
    decodebin の src pad を nvstreammux の sink_0 にリンク
    data = {"streammux": <nvstreammux>}
    """
    caps = pad.get_current_caps()
    if not caps:
        return
    name = caps.get_structure(0).get_name()
    # 映像のみ拾う
    if name.startswith("video/"):
        streammux = data["streammux"]
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            LOG.error("nvstreammux sink pad 取得失敗")
            return
        if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            LOG.error("decodebin→nvstreammux リンク失敗")
        else:
            LOG.info("映像を nvstreammux に接続しました")

def bus_call(bus, message, loop, pipeline):
    t = message.type
    if t == Gst.MessageType.EOS:
        LOG.info("EOS")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        LOG.error(f"ERROR: {err} ({debug})")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        LOG.warning(f"WARNING: {warn} ({debug})")
    elif t == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
        old, new, pending = message.parse_state_changed()
        LOG.info(f"Pipeline state: {old.value_nick} -> {new.value_nick}")
    return True

def build_pipeline(args: argparse.Namespace) -> Gst.Pipeline:
    """
    uridecodebin
      -> nvstreammux(batch=1, width/height/fps設定)
      -> nvinfer(primary=face)
      -> dsexample(process-mode=2=objects, blur-objects=1, mosaic-level=*)
      -> nvvideoconvert
      -> caps (NV12)
      -> nvv4l2h264enc
      -> h264parse
      -> flvmux (audioも合流)
      -> rtmpsink
    """
    pipe = Gst.Pipeline.new("rtsp-youtube-face-mosaic")
    if not pipe:
        raise RuntimeError("Pipeline作成失敗")

    # Source
    source = make("uridecodebin", "rtsp-source", uri=args.rtsp_url)
    # DeepStream バッチャ
    streammux = make("nvstreammux", "stream-muxer",
                     batch_size=1,
                     width=args.width,
                     height=args.height,
                     enable-padding=False,
                     live_source=True,
                     buffer_pool_size=8)
    # 推論（顔）
    pgie = make("nvinfer", "primary-fd",
                config_file_path=args.infer_config,
                unique_id=1)

    # モザイク（dsexample）
    # process-mode=2: オブジェクト単位処理、blur-objects=1: 検出枠にモザイク
    # mosaic-level: モザイク粒度（数値大きいほど荒くなる）
    dsex = make("dsexample", "mosaic",
                full_frame=0,
                process_mode=2,
                blur_objects=1,
                mosaic_size=args.mosaic_level,
                unique_id=15)  # 任意

    # NV12へ（エンコード前）
    nvvidconv = make("nvvideoconvert", "nvvidconv-post")
    capsfilter = make("capsfilter", "caps-nv12",
                      caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    # H.264 HW エンコード → FLV → RTMP
    encoder = make("nvv4l2h264enc", "hw-encoder",
                   bitrate=args.bitrate,
                   insert_sps_pps=True,
                   iframeinterval=args.fps,  # 1秒毎にIDR
                   preset_level=1,            # default
                   profile=0)                 # Baseline
    h264parse = make("h264parse", "h264-parser")
    q_video = make("queue", "q-video")
    flvmux = make("flvmux", "flv-muxer", streamable=True)
    sink = make("rtmpsink", "youtube-sink", location=args.youtube_url)

    # ダミー音声（無音AAC）
    atsrc = make("audiotestsrc", "audio-src", wave=4, is_live=True)
    aconv = make("audioconvert", "audio-conv")
    ares = make("audioresample", "audio-res")
    aenc = make("voaacenc", "aac-enc", bitrate=128000)
    aparse = make("aacparse", "aac-parse")
    q_audio = make("queue", "q-audio")

    for e in (source, streammux, pgie, dsex, nvvidconv, capsfilter,
              encoder, h264parse, q_video, flvmux, sink,
              atsrc, aconv, ares, aenc, aparse, q_audio):
        pipe.add(e)

    # source は動的リンク
    source.connect("pad-added", on_pad_added_decodebin, {"streammux": streammux})

    # DeepStream line
    assert streammux.link(pgie)
    assert pgie.link(dsex)
    assert dsex.link(nvvidconv)
    assert nvvidconv.link(capsfilter)
    assert capsfilter.link(encoder)
    assert encoder.link(h264parse)
    assert h264parse.link(q_video)

    # video → flvmux
    vpad = flvmux.get_request_pad("video")
    if q_video.get_static_pad("src").link(vpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("video→flvmux接続失敗")

    # audio line
    assert atsrc.link(aconv)
    assert aconv.link(ares)
    assert ares.link(aenc)
    assert aenc.link(aparse)
    assert aparse.link(q_audio)

    # audio → flvmux
    apad = flvmux.get_request_pad("audio")
    if q_audio.get_static_pad("src").link(apad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("audio→flvmux接続失敗")

    # flvmux → RTMP
    assert flvmux.link(sink)

    LOG.info("パイプライン構築完了（DeepStream顔モザイク）")
    return pipe

def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    Gst.init(None)
    pipe = build_pipeline(args)
    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop, pipe)

    def handle_sig(signum, frame):
        LOG.info(f"Signal {signum} 受信、停止します")
        loop.quit()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        LOG.info(f"配信開始 -> {args.youtube_url}")
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            LOG.error("再生開始に失敗")
            return 1
        loop.run()
    finally:
        LOG.info("停止処理中…")
        pipe.set_state(Gst.State.NULL)
    return 0

if __name__ == "__main__":
    sys.exit(main())
