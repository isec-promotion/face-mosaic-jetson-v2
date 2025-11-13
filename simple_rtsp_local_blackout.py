#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSP → nvurisrcbin(HWデコード) → nvstreammux → nvinfer(顔検出) → 黒塗り → ローカル表示
- DeepStream 標準要素のみ（OpenCV不要）
- 「黒塗り」は nvdsosd の前段 pad-probe で bbox を塗りつぶし
- H.264/H.265 を自動判別（nvurisrcbin）
例:
python3 simple_rtsp_local_blackout.py \
  "rtsp://user:pass@IP:554/Streaming/Channels/101" \
  --infer-config ./config_infer_primary_face_yolo11.txt \
  --width 1920 --height 1080 --fps 30 --tcp --log-level INFO
"""
import argparse
import logging
import signal
import sys

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib
import pyds

LOG = logging.getLogger("rtsp-local-face-blackout")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30

def parse_args():
    p = argparse.ArgumentParser(description="RTSP→顔黒塗り→ローカル表示（DeepStream）")
    p.add_argument("rtsp_url", help="rtsp://user:pass@ip:554/Streaming/Channels/101")
    p.add_argument("--infer-config", default="./config_infer_primary_face_yolo11.txt",
                   help="nvinfer の設定ファイル（顔検出）")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--latency", type=int, default=200, help="RTSP latency(ms)")
    p.add_argument("--tcp", action="store_true", help="RTSPをTCPで受信（既定はUDP）")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def make(factory: str, name: str, **props) -> Gst.Element:
    e = Gst.ElementFactory.make(factory, name)
    if not e:
        raise RuntimeError(f"要素作成失敗: {factory} ({name})")
    for k, v in props.items():
        try:
            e.set_property(k, v)
        except TypeError:
            pass
    return e

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
        old, new, _ = message.parse_state_changed()
        LOG.info(f"Pipeline: {old.value_nick} -> {new.value_nick}")
    return True

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """nvdsosd の前で、検出BBoxを黒塗りに変更"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            rp = obj_meta.rect_params
            rp.has_bg_color = 1
            rp.bg_color.set(0.0, 0.0, 0.0, 1.0)  # RGBA 黒・不透明
            rp.border_width = 0
            rp.border_color.set(0.0, 0.0, 0.0, 0.0)

            l_obj = l_obj.next
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

def build_pipeline(args: argparse.Namespace) -> Gst.Pipeline:
    Gst.init(None)
    pipe = Gst.Pipeline.new("rtsp-face-blackout-local")
    if not pipe:
        raise RuntimeError("Pipeline作成失敗")

    # --- 入力：nvurisrcbin（H.264/H.265自動対応・再接続） ---
    # 参考プロパティ:
    #  - uri: RTSP/HTTP(S)など一括指定
    #  - latency: RTSPジッタバッファ
    #  - rtsp-reconnect-interval-sec: 切断時の自動再接続間隔
    #  - enable-udpsrc-buffer-size/udp-buffer-size: 大きめに
    #  - cudadec-memtype: 0(NVMM)
    #  - drop-on-latency: Trueで遅延時ドロップ
    src = make(
        "nvurisrcbin", "src",
        uri=args.rtsp_url,
        latency=args.latency,
        drop_on_latency=True,
        rtsp_reconnect_interval_sec=5,
        rtsp_reconnect_attempts=-1,
        enable_udpsrc_buffer_size=True,
        udp_buffer_size=1<<20,   # 1MB
        cudadec_memtype=0
    )
    # TCP指定（プロパティが存在する環境のみ反映）
    if args.tcp:
        try:
            src.set_property("rtsp_transport", 0)  # 0:TCP, 1:UDP (環境により異なる場合あり)
        except Exception:
            pass

    # 前処理（NVMMでNV12に統一）
    nvconv_pre = make("nvvideoconvert", "nvconv-pre")
    caps_pre   = make("capsfilter", "caps-pre",
                      caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    mux = make("nvstreammux", "mux",
               batch_size=1, width=args.width, height=args.height,
               live_source=True, buffer_pool_size=8)

    pgie = make("nvinfer", "pgie",
                config_file_path=args.infer_config, unique_id=1)

    osd = make("nvdsosd", "osd",
               process_mode=0,   # GPU
               display_text=0)   # ラベル非表示

    nvconv = make("nvvideoconvert", "nvconv")
    xform  = make("nvegltransform", "xform")
    sink   = make("nveglglessink", "sink", sync=False)

    for e in (src, nvconv_pre, caps_pre, mux, pgie, osd, nvconv, xform, sink):
        pipe.add(e)

    # nvurisrcbin → nvvideoconvert → caps → nvstreammux.sink_0
    assert src.link(nvconv_pre)
    assert nvconv_pre.link(caps_pre)

    sinkpad_mux = mux.request_pad_simple("sink_0")
    if caps_pre.get_static_pad("src").link(sinkpad_mux) != Gst.PadLinkReturn.OK:
        raise RuntimeError("caps_pre → mux.sink_0 リンク失敗")

    assert mux.link(pgie)
    assert pgie.link(osd)
    assert osd.link(nvconv)
    assert nvconv.link(xform)
    assert xform.link(sink)

    # 黒塗りを OSD の手前に差し込む
    osd_sink_pad = osd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("nvdsosd sink pad 取得失敗")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    LOG.info("パイプライン構築完了（nvurisrcbin→黒塗り→ローカル表示）")
    return pipe

def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    pipe = build_pipeline(args)
    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop, pipe)

    def handle_sig(signum, frame):
        LOG.info(f"Signal {signum} を受信、停止")
        loop.quit()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        LOG.info("再生開始")
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
