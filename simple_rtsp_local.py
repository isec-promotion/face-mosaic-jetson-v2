#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSP → 顔モザイク（GPU / DeepStream）→ 画面表示（ローカル確認用）
- NVIDIA Jetson / DeepStream 要素のみ
- nvinfer + dsexample（モザイク）
- CPUに降ろさず NVMM 上で処理
python3 simple_rtsp_local.py \
  "rtsp://user:pass@IP:554/Streaming/Channels/101" \
  --infer-config ./config_infer_primary_facedet.txt \
  --mosaic-level 6 \
  --width 1920 --height 1080 --fps 30
"""

import argparse
import logging
import signal
import sys

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

LOG = logging.getLogger("rtsp-local-face-mosaic-ds")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTSP→顔モザイク→ローカル表示（DeepStream最小構成）")
    p.add_argument("rtsp_url", help="RTSP URL（例: rtsp://user:pass@ip:554/Streaming/Channels/101）")
    p.add_argument("--infer-config", default="./config_infer_primary_facedet.txt",
                   help="nvinfer の設定ファイル（顔検出モデルを指定）")
    p.add_argument("--mosaic-level", type=int, default=6,
                   help="モザイクの粒度（3〜10程度。大きいほど荒い）")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="処理基準の幅（nvstreammux）")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="処理基準の高さ（nvstreammux）")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="処理基準のFPS（nvstreammux）")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()

def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def make(factory: str, name: str, **props) -> Gst.Element:
    e = Gst.ElementFactory.make(factory, name)
    if not e:
        raise RuntimeError(f"GStreamer 要素作成失敗: {factory} ({name})")
    for k, v in props.items():
        e.set_property(k, v)
    return e

def on_pad_added_decodebin(decodebin, pad, data):
    """uridecodebin の新規 pad を nvstreammux の sink_0 に接続"""
    caps = pad.get_current_caps()
    if not caps:
        return
    name = caps.get_structure(0).get_name()
    if name.startswith("video/"):
        streammux = data["streammux"]
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            LOG.error("nvstreammux sink pad 取得失敗")
            return
        if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            LOG.error("decodebin→nvstreammux のリンク失敗")
        else:
            LOG.info("映像を nvstreammux に接続しました")

def bus_call(bus, message, loop, pipeline):
    t = message.type
    if t == Gst.MessageType.EOS:
        LOG.info("EOS 受信。終了します。")
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
        LOG.info(f"Pipeline: {old.value_nick} -> {new.value_nick}")
    return True

def build_pipeline(args: argparse.Namespace) -> Gst.Pipeline:
    """
    パイプライン:
      uridecodebin
        -> nvstreammux(batch=1, width/height/fps)
        -> nvinfer(顔検出; config_infer_primary_facedet.txt流用)
        -> dsexample(検出枠にモザイク)
        -> nvvideoconvert
        -> nvegltransform
        -> nveglglessink（Jetson画面表示）
    すべて NVMM 上で処理し、エンコードは行わない（ローカル確認用）。
    """
    pipe = Gst.Pipeline.new("rtsp-face-mosaic-local")
    if not pipe:
        raise RuntimeError("Pipeline 作成失敗")

    # 1) ソース（RTSP）
    source = make("uridecodebin", "rtsp-source", uri=args.rtsp_url)
    # NG: uridecodebin には live プロパティなし
    # source.set_property("live", True)

    # 2) バッチャ
    streammux = make("nvstreammux", "stream-muxer",
                     batch_size=1,
                     width=args.width,
                     height=args.height,
                     live_source=True,
                     buffer_pool_size=8)

    # 3) 推論（顔）
    pgie = make("nvinfer", "primary-fd",
                config_file_path=args.infer_config,
                unique_id=1)

    # 4) モザイク（dsexample）
    #   process-mode=2: オブジェクト単位処理
    #   blur-objects=1: 検出BBOXをモザイク（ブラーでなくモザイク）
    dsex = make("dsexample", "mosaic",
                full_frame=0,
                process_mode=2,
                blur_objects=1,
                mosaic_size=args.mosaic_level,
                unique_id=15)

    # 5) 表示前の変換
    nvvidconv = make("nvvideoconvert", "nvvidconv")
    # EGL へ渡す前段
    nveglxform = make("nvegltransform", "nveglxform")

    # 6) 画面表示
    #   Jetson では nveglglessink が最も相性良い
    sink = make("nveglglessink", "display", sync=False)  # 遅延を抑制するため sync=False が無難

    for e in (source, streammux, pgie, dsex, nvvidconv, nveglxform, sink):
        pipe.add(e)

    # uridecodebin は動的リンク
    source.connect("pad-added", on_pad_added_decodebin, {"streammux": streammux})

    # DeepStream ラインを静的リンク
    assert streammux.link(pgie)
    assert pgie.link(dsex)
    assert dsex.link(nvvidconv)
    assert nvvidconv.link(nveglxform)
    assert nveglxform.link(sink)

    LOG.info("パイプライン構築完了（ローカル表示・顔モザイク）")
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
        LOG.info(f"Signal {signum} を受信。停止します。")
        loop.quit()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        LOG.info("再生開始")
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            LOG.error("再生開始に失敗しました")
            return 1
        loop.run()
    finally:
        LOG.info("停止処理中…")
        pipe.set_state(Gst.State.NULL)
    return 0

if __name__ == "__main__":
    sys.exit(main())
