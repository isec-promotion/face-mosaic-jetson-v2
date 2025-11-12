#!/usr/bin/env python3
"""
使い方:
    python3 deepstream_youtube.py "<RTSP_URL>" "<YOUTUBE_STREAM_KEY>"

Jetson Orin NX Super向けにDeepStreamでRTSP監視映像を取得し、顔をマスクしてYouTube Liveへ中継するスクリプト。

フレームは常にNVMM/GPUメモリ内で処理されるゼロコピー構成です:
uridecodebin（HWデコード）-> nvstreammux -> nvinfer -> nvosd（塗りつぶしマスク）->
nvvideoconvert -> nvv4l2h264enc -> flvmux -> rtmpsink
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib  # noqa: E402

import pyds  # noqa: E402


LOG = logging.getLogger("deepstream-face-streamer")

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_BITRATE = 2_500_000  # 2.5 Mbps相当


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest an RTSP surveillance stream, mask faces entirely on the GPU, "
            "then push the result to YouTube Live (RTMPS)."
        )
    )
    parser.add_argument(
        "rtsp_url",
        help="RTSP URL from the surveillance camera "
        "(e.g. rtsp://user:pass@ip:554/Streaming/Channels/101)",
    )
    parser.add_argument(
        "stream_key",
        help="YouTube Live stream key copied from the Live Control Room",
    )
    parser.add_argument(
        "--infer-config",
        default="config_infer_primary_facedetect.txt",
        help="Path to the nvinfer configuration file",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Output width in pixels (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output height in pixels (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Output FPS (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=DEFAULT_BITRATE,
        help=f"H.264 encoder bitrate (default: {DEFAULT_BITRATE})",
    )
    parser.add_argument(
        "--youtube-ingest",
        default="rtmps://a.rtmps.youtube.com:443/live2",
        help="YouTube RTMPS ingest endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
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
        raise RuntimeError(f"Failed to create GStreamer element '{factory}' ({name})")
    for key, value in properties.items():
        element.set_property(key, value)
    return element


def on_decodebin_pad_added(decodebin, pad, data):
    """decodebinの出力をnvstreammuxの入力パッドへ接続する。"""
    sinkpad = data.get_request_pad("sink_0")
    if not sinkpad:
        LOG.error("Unable to get sink pad from streammux")
        return

    caps = pad.get_current_caps()
    caps_struct = caps.get_structure(0)
    caps_name = caps_struct.get_name()

    if "video" not in caps_name:
        LOG.debug("Ignoring non-video pad: %s", caps_name)
        data.release_request_pad(sinkpad)
        return

    if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        LOG.error("Failed to link decodebin to streammux")
    else:
        LOG.info("Linked decodebin to streammux")


def decodebin_child_added(child_proxy, object, name):
    """利用可能な場合にハードウェアデコード経路を強制する。"""
    LOG.debug("decodebin child added: %s (%s)", name, object.__gtype__.name)
    if name.startswith("decodebin") or name.startswith("nvv4l2decoder"):
        object.set_property("drop-on-late", True)


def osd_sink_pad_buffer_probe(pad, info, _u_data):
    """検出済みの顔をGPU上で塗りつぶし矩形に変換する。"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        LOG.warning("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            if obj_meta.class_id == 0:
                rect_params = obj_meta.rect_params
                rect_params.border_width = 0
                rect_params.has_bg_color = 1
                rect_params.bg_color.set(0.0, 0.0, 0.0, 1.0)
                text_params = obj_meta.text_params
                text_params.display_text = ""
                text_params.set_bg_clr = 0
                text_params.set_font_clr = 0
            l_obj = l_obj.next
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


def bus_call(bus, message, loop):
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        LOG.info("End-of-stream received")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        LOG.error("Error: %s (%s)", err, debug)
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        LOG.warning("Warning: %s (%s)", warn, debug)
    return True


def validate_files(args: argparse.Namespace) -> None:
    infer_config = Path(args.infer_config)
    if not infer_config.is_file():
        raise FileNotFoundError(
            f"Infer config not found: {infer_config.resolve()}"
        )


def build_pipeline(args: argparse.Namespace) -> Gst.Pipeline:
    pipeline = Gst.Pipeline.new("face-mosaic-pipeline")
    if not pipeline:
        raise RuntimeError("Failed to create pipeline")

    streammux = make_element("nvstreammux", "stream-muxer")
    streammux.set_property("width", args.width)
    streammux.set_property("height", args.height)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", int(1e6 / args.fps))
    streammux.set_property("live-source", 1)

    source = make_element("uridecodebin", "rtsp-source", uri=args.rtsp_url)
    source.connect("pad-added", on_decodebin_pad_added, streammux)
    source.connect("child-added", decodebin_child_added)

    pgie = make_element("nvinfer", "primary-inference", config_file_path=args.infer_config)

    nvosd = make_element("nvdsosd", "on-screen-display", process_mode=1, display_text=0)

    nvvidconv = make_element("nvvideoconvert", "nv-video-converter")
    capsfilter = make_element(
        "capsfilter",
        "caps-filter",
        caps=Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12, framerate={}/1".format(args.fps)
        ),
    )

    encoder = make_element(
        "nvv4l2h264enc",
        "hw-encoder",
        bitrate=args.bitrate,
        control_rate=1,
        iframeinterval=args.fps * 2,
        preset_level=1,
        insert_sps_pps=True,
        maxperf_enable=True,
        EnableTwopassCBR=False,
    )
    h264parser = make_element("h264parse", "h264-parser")
    flvmux = make_element("flvmux", "flv-muxer", streamable=True)
    sink = make_element("rtmpsink", "youtube-sink", **{
        "location": args.youtube_url,
        "sync": False,
        "async": False,
    })

    for elem in (
        source,
        streammux,
        pgie,
        nvosd,
        nvvidconv,
        capsfilter,
        encoder,
        h264parser,
        flvmux,
        sink,
    ):
        pipeline.add(elem)

    if not Gst.Element.link_many(
        streammux,
        pgie,
        nvosd,
        nvvidconv,
        capsfilter,
        encoder,
        h264parser,
        flvmux,
        sink,
    ):
        raise RuntimeError("Failed to link elements")

    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("Unable to get OSD sink pad")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    return pipeline


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    validate_files(args)

    Gst.init(None)

    pipeline = build_pipeline(args)
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    def handle_sigint(signum, frame):
        LOG.info("Signal %s received, stopping...", signum)
        loop.quit()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        LOG.info("Starting pipeline -> %s", args.youtube_url)
        pipeline.set_state(Gst.State.PLAYING)
        loop.run()
    finally:
        LOG.info("Stopping pipeline")
        pipeline.set_state(Gst.State.NULL)

    return 0


if __name__ == "__main__":
    sys.exit(main())
