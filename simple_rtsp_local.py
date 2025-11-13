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

gst-inspect-1.0 dsexample | sed -n '/Properties:/,/Pad Templates:/p'
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

def parse_args():
    p = argparse.ArgumentParser(description="RTSP→顔モザイク→ローカル表示（DeepStream）")
    p.add_argument("rtsp_url", help="rtsp://user:pass@ip:554/Streaming/Channels/101 等")
    p.add_argument("--infer-config", default="./config_infer_primary_facedet.txt",
                   help="nvinfer の設定ファイル")
    p.add_argument("--mosaic-level", type=int, default=6,  # 予備：将来拡張用（未使用）
                   help="（一部の dsexample でモザイク粒度を変えられる場合に使用）")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()

def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def make(factory: str, name: str, **props) -> Gst.Element:
    e = Gst.ElementFactory.make(factory, name)
    if not e:
        raise RuntimeError(f"要素作成失敗: {factory} ({name})")
    for k, v in props.items():
        e.set_property(k, v)
    return e

def on_pad_added_decodebin(decodebin, pad, data):
    """
    decodebin -> (nvvideoconvert_pre -> caps_pre) -> nvstreammux.sink_0
    """
    caps = pad.get_current_caps()
    if not caps:
        return
    name = caps.get_structure(0).get_name()
    if not name.startswith("video/"):
        return

    nvvidconv_pre: Gst.Element = data["nvvidconv_pre"]
    caps_pre: Gst.Element = data["caps_pre"]
    streammux: Gst.Element = data["streammux"]

    # decodebin → nvvideoconvert(pre)
    sinkpad_conv = nvvidconv_pre.get_static_pad("sink")
    if sinkpad_conv and not sinkpad_conv.is_linked():
        if pad.link(sinkpad_conv) != Gst.PadLinkReturn.OK:
            LOG.error("decodebin → nvvideoconvert(pre) のリンク失敗")
            return
        LOG.info("decodebin → nvvideoconvert(pre) 接続")

    # caps_pre → streammux.sink_0
    srcpad_caps = caps_pre.get_static_pad("src")
    sinkpad_mux = streammux.get_request_pad("sink_0")
    if not sinkpad_mux:
        LOG.error("nvstreammux sink_0 取得失敗")
        return
    if srcpad_caps.is_linked():
        return
    if srcpad_caps.link(sinkpad_mux) != Gst.PadLinkReturn.OK:
        LOG.error("caps_pre → nvstreammux.sink_0 のリンク失敗")
    else:
        LOG.info("caps_pre → nvstreammux.sink_0 接続")

def bus_call(bus, message, loop, pipeline):
    t = message.type
    if t == Gst.MessageType.EOS:
        LOG.info("EOS 受信。終了。")
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
    uridecodebin
      -> nvvideoconvert(pre)
      -> caps(memory:NVMM,NV12)
      -> nvstreammux(batch=1)
      -> nvinfer(顔)
      -> dsexample(検出枠をモザイク/ブラー)
      -> nvvideoconvert
      -> nvegltransform
      -> nveglglessink
    """
    pipe = Gst.Pipeline.new("rtsp-face-mosaic-local")
    if not pipe:
        raise RuntimeError("Pipeline 作成失敗")

    # Source
    source = make("uridecodebin", "rtsp-source", uri=args.rtsp_url)

    # pre: CPU→NVMM 変換
    nvvidconv_pre = make("nvvideoconvert", "nvvidconv-pre")
    caps_pre = make("capsfilter", "caps-pre",
                    caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    # DeepStream core
    streammux = make("nvstreammux", "stream-muxer",
                     batch_size=1,
                     width=args.width,
                     height=args.height,
                     live_source=True,
                     buffer_pool_size=8)
    pgie = make("nvinfer", "primary-fd",
                config_file_path=args.infer_config,
                unique_id=1)

    # dsexample（確実に存在するプロパティのみ設定）
    dsex = make("dsexample", "mosaic")
    dsex.set_property("full-frame", 0)     # 0: オブジェクト領域のみ処理
    dsex.set_property("blur-objects", 1)   # 1: 検出枠をモザイク/ブラー（実装依存）

    # 表示系
    nvvidconv = make("nvvideoconvert", "nvvidconv")
    nveglxform = make("nvegltransform", "nveglxform")
    sink = make("nveglglessink", "display", sync=False)

    for e in (source, nvvidconv_pre, caps_pre,
              streammux, pgie, dsex, nvvidconv, nveglxform, sink):
        pipe.add(e)

    # 動的リンク: decodebin → (pre) → streammux
    source.connect("pad-added", on_pad_added_decodebin, {
        "nvvidconv_pre": nvvidconv_pre,
        "caps_pre": caps_pre,
        "streammux": streammux
    })

    # 静的リンク（pre）
    assert nvvidconv_pre.link(caps_pre)

    # DS line
    assert streammux.link(pgie)
    assert pgie.link(dsex)
    assert dsex.link(nvvidconv)
    assert nvvidconv.link(nveglxform)
    assert nveglxform.link(sink)

    LOG.info("パイプライン構築完了（ローカル表示・顔モザイク / 汎用プロパティ）")
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
