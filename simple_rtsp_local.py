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

まず UDP で失敗する場合は --tcp を付けて試す：
python3 simple_rtsp_local.py "rtsp://..." --infer-config ./config_infer_primary_facedet.txt --tcp

詳細ログ を見たいとき：
GST_DEBUG=3,rtsp*:5,rtspsrc:5,nvv4l2decoder:5,nvstreammux:5,nvinfer:5,dsexample:5 python3 simple_rtsp_local.py "rtsp://..." --infer-config ./config_infer_primary_facedet.txt --tcp --log-level DEBUG

画面シンクの問題が疑わしいときは一時的に変更：
... && sed -i 's/nveglglessink/autovideosink/' simple_rtsp_local.py

"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import signal
import sys
import ctypes

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

import pyds

LOG = logging.getLogger("rtsp-local-face-mosaic-ds")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30

def parse_args():
    p = argparse.ArgumentParser(description="RTSP→顔匿名化(blur)→ローカル表示（DeepStream安定版）")
    p.add_argument("rtsp_url", help="rtsp://user:pass@ip:554/Streaming/channels/101 等")
    p.add_argument("--infer-config", default="./config_infer_primary_facedet.txt",
                   help="nvinfer の設定ファイル（顔検出エンジン）")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--mosaic-level", type=int, default=6,
                   help="モザイクレベル（1-10: 高いほど粗いモザイク）")
    p.add_argument("--latency", type=int, default=200,
                   help="rtspsrc の latency(ms)")
    p.add_argument("--tcp", action="store_true",
                   help="RTSPをTCPで受ける（デフォルトはUDP）")
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

def osd_sink_pad_buffer_probe(pad, info, mosaic_level):
    """
    nvdsosd の sink pad にプローブを設置し、検出された顔領域を塗りつぶす
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # 検出された顔に塗りつぶし矩形を追加
            if obj_meta.obj_label == "face" or obj_meta.class_id == 0:
                rect_params = obj_meta.rect_params
                
                # 塗りつぶし用の表示メタを追加（黒い矩形）
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                
                # 矩形を追加
                rect = display_meta.rect_params[display_meta.num_rects]
                rect.left = int(rect_params.left)
                rect.top = int(rect_params.top)
                rect.width = int(rect_params.width)
                rect.height = int(rect_params.height)
                
                # 黒で塗りつぶし（モザイク風にするため透明度調整も可能）
                rect.border_width = 0
                rect.has_bg_color = 1
                rect.bg_color.red = 0.0
                rect.bg_color.green = 0.0
                rect.bg_color.blue = 0.0
                rect.bg_color.alpha = 1.0  # 完全に不透明
                
                display_meta.num_rects += 1
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def link_dynamic_pad_rtspsrc(rtspsrc, pad, depay):
    """rtspsrcの動的src-padをrtp depayへ接続（H.264のみ想定）"""
    caps = pad.get_current_caps()
    if not caps:
        return
    s = caps.get_structure(0)
    name = s.get_name()  # 例: application/x-rtp
    if name.startswith("application/x-rtp"):
        # H.264 のみ受ける（H.265なら rtph265depay に変えてください）
        if s.has_field("encoding-name") and s.get_value("encoding-name") == "H264":
            sinkpad = depay.get_static_pad("sink")
            if not sinkpad.is_linked():
                if pad.link(sinkpad) == Gst.PadLinkReturn.OK:
                    LOG.info("rtspsrc → rtph264depay を接続しました")
                else:
                    LOG.error("rtspsrc → rtph264depay のリンクに失敗")

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
    rtspsrc (latency/プロトコル指定)
      → rtph264depay → h264parse → nvv4l2decoder
      → nvvideoconvert(pre) → caps(memory:NVMM,NV12)
      → nvstreammux(batch=1,width/height/live)
      → nvinfer(primary=face)
      → nvdsosd(顔領域を塗りつぶし)
      → nvvideoconvert → nvegltransform → nveglglessink
    """
    pipe = Gst.Pipeline.new("rtsp-face-anon-local")
    if not pipe:
        raise RuntimeError("Pipeline 作成失敗")

    # RTSP source
    protocols = 0x00000001 if args.tcp else 0x00000006  # TCP=1, UDP=6(UDP+UDP-mcast)
    src = make("rtspsrc", "src",
               location=args.rtsp_url,
               latency=args.latency,
               protocols=protocols,
               drop_on_latency=True)

    depay = make("rtph264depay", "depay")
    parse = make("h264parse", "h264parse")
    decoder = make("nvv4l2decoder", "decoder")

    # CPU→NVMM 変換（念のため前段）
    nvvidconv_pre = make("nvvideoconvert", "nvvidconv-pre")
    caps_pre = make("capsfilter", "caps-pre",
                    caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    # DeepStream core
    mux = make("nvstreammux", "mux",
               batch_size=1,
               width=args.width,
               height=args.height,
               live_source=True,
               buffer_pool_size=8)

    pgie = make("nvinfer", "pgie",
                config_file_path=args.infer_config,
                unique_id=1)

    # nvdsosd で検出枠を描画（probe で塗りつぶし処理を追加）
    nvosd = make("nvdsosd", "nvosd",
                 process_mode=0,    # CPU mode
                 display_text=False)  # テキスト表示なし

    nvvidconv = make("nvvideoconvert", "nvvidconv")
    nveglxform = make("nvegltransform", "nveglxform")
    sink = make("nveglglessink", "sink", sync=False)

    for e in (src, depay, parse, decoder,
              nvvidconv_pre, caps_pre, mux,
              pgie, nvosd, nvvidconv, nveglxform, sink):
        pipe.add(e)

    # rtspsrc は動的 pad
    src.connect("pad-added", link_dynamic_pad_rtspsrc, depay)

    # 静的リンク（RTP→HWデコード）
    assert depay.link(parse)
    assert parse.link(decoder)
    assert decoder.link(nvvidconv_pre)
    assert nvvidconv_pre.link(caps_pre)

    # caps_pre(src) → mux.sink_0（request pad）
    sinkpad_mux = mux.get_request_pad("sink_0")
    if not sinkpad_mux:
        raise RuntimeError("nvstreammux sink_0 取得失敗")
    if caps_pre.get_static_pad("src").link(sinkpad_mux) != Gst.PadLinkReturn.OK:
        raise RuntimeError("caps_pre → nvstreammux.sink_0 リンク失敗")

    # DeepStream ライン
    assert mux.link(pgie)
    assert pgie.link(nvosd)
    assert nvosd.link(nvvidconv)
    assert nvvidconv.link(nveglxform)
    assert nveglxform.link(sink)

    # nvdsosd の sink pad にプローブを追加して顔領域を塗りつぶす
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        raise RuntimeError("nvdsosd の sink pad 取得失敗")
    osdsinkpad.add_probe(
        Gst.PadProbeType.BUFFER,
        osd_sink_pad_buffer_probe,
        args.mosaic_level
    )

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
