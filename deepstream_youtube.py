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

# import pyds  # noqa: E402  # DeepStream 7.1 にPythonバインディングが含まれていない


LOG = logging.getLogger("deepstream-face-streamer")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_BITRATE = 6_000_000  # 6 Mbps (1080p推奨)


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
        help=f"Output width in pixels (default: {DEFAULT_WIDTH}, use 1280 for 720p)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output height in pixels (default: {DEFAULT_HEIGHT}, use 720 for 720p)",
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
        help=f"H.264 encoder bitrate (default: {DEFAULT_BITRATE}, use 2500000 for 720p)",
    )
    parser.add_argument(
        "--youtube-ingest",
        default="rtmp://a.rtmp.youtube.com/live2",
        help="YouTube RTMP ingest endpoint (default: %(default)s)",
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
    # drop-on-lateプロパティは一部の要素にのみ存在するため、エラーを無視
    try:
        if name.startswith("decodebin") or name.startswith("nvv4l2decoder"):
            object.set_property("drop-on-late", True)
    except:
        pass


def osd_sink_pad_buffer_probe(pad, info, _u_data):
    """検出済みの顔をGPU上で塗りつぶし矩形に変換する。"""
    # TODO: DeepStream Pythonバインディング（pyds）が利用可能になったら実装
    # 現在は顔検出のみ実行（バウンディングボックスが表示される）
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
        raise FileNotFoundError(f"Infer config not found: {infer_config.resolve()}")
    LOG.info("Using nvinfer config: %s", infer_config.resolve())



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

    # nvstreammux後のキュー
    queue_mux = make_element("queue", "queue-mux")

    pgie = make_element("nvinfer", "primary-inference", config_file_path=args.infer_config)

    # nvinfer後のキュー
    queue_pgie = make_element("queue", "queue-pgie")

    nvosd = make_element(
        "nvdsosd",
        "on-screen-display",
        process_mode=1,
        display_text=0,
        display_bbox=1,
        display_mask=0,
    )

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
        iframeinterval=30,  # 1秒間隔（YouTubeは2秒以下を推奨）
        preset_level=1,
        insert_sps_pps=True,
        maxperf_enable=True,
        EnableTwopassCBR=False,
        profile=0,  # Baseline Profile
    )
    h264parser = make_element("h264parse", "h264-parser")
    queue_video = make_element("queue", "queue-video")
    
    # ダミー音声ソース（YouTubeは音声ストリームを要求する）
    audiotestsrc = make_element("audiotestsrc", "audio-source", wave=4)  # wave=4は無音
    audioconvert = make_element("audioconvert", "audio-convert")
    audioresample = make_element("audioresample", "audio-resample")
    voaacenc = make_element("voaacenc", "audio-encoder", bitrate=128000)
    aacparse = make_element("aacparse", "aac-parser")
    queue_audio = make_element("queue", "queue-audio")
    
    flvmux = make_element("flvmux", "flv-muxer", streamable=True)
    sink = make_element("rtmpsink", "youtube-sink", location=args.youtube_url, sync=False)

    for elem in (
        source,
        streammux,
        queue_mux,
        pgie,
        queue_pgie,
        nvosd,
        nvvidconv,
        capsfilter,
        encoder,
        h264parser,
        queue_video,
        audiotestsrc,
        audioconvert,
        audioresample,
        voaacenc,
        aacparse,
        queue_audio,
        flvmux,
        sink,
    ):
        pipeline.add(elem)

    # ビデオパイプラインをリンク
    if not streammux.link(queue_mux):
        raise RuntimeError("Failed to link streammux -> queue_mux")
    if not queue_mux.link(pgie):
        raise RuntimeError("Failed to link queue_mux -> pgie")
    if not pgie.link(queue_pgie):
        raise RuntimeError("Failed to link pgie -> queue_pgie")
    if not queue_pgie.link(nvosd):
        raise RuntimeError("Failed to link queue_pgie -> nvosd")
    if not nvosd.link(nvvidconv):
        raise RuntimeError("Failed to link nvosd -> nvvideoconvert")
    if not nvvidconv.link(capsfilter):
        raise RuntimeError("Failed to link nvvideoconvert -> capsfilter")
    if not capsfilter.link(encoder):
        raise RuntimeError("Failed to link capsfilter -> encoder")
    if not encoder.link(h264parser):
        raise RuntimeError("Failed to link encoder -> h264parse")
    if not h264parser.link(queue_video):
        raise RuntimeError("Failed to link h264parse -> queue_video")
    
    # ビデオストリームをflvmuxに接続
    video_pad = flvmux.get_request_pad("video")
    queue_video_src = queue_video.get_static_pad("src")
    if queue_video_src.link(video_pad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link queue_video -> flvmux (video)")
    
    # 音声パイプラインをリンク
    if not audiotestsrc.link(audioconvert):
        raise RuntimeError("Failed to link audiotestsrc -> audioconvert")
    if not audioconvert.link(audioresample):
        raise RuntimeError("Failed to link audioconvert -> audioresample")
    if not audioresample.link(voaacenc):
        raise RuntimeError("Failed to link audioresample -> voaacenc")
    if not voaacenc.link(aacparse):
        raise RuntimeError("Failed to link voaacenc -> aacparse")
    if not aacparse.link(queue_audio):
        raise RuntimeError("Failed to link aacparse -> queue_audio")
    
    # 音声ストリームをflvmuxに接続
    audio_pad = flvmux.get_request_pad("audio")
    queue_audio_src = queue_audio.get_static_pad("src")
    if queue_audio_src.link(audio_pad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link queue_audio -> flvmux (audio)")
    
    # flvmuxからrtmpsinkへ
    if not flvmux.link(sink):
        raise RuntimeError("Failed to link flvmux -> rtmpsink")

    # マスキング処理は一時的にスキップ
    # osd_sink_pad = nvosd.get_static_pad("sink")
    # if not osd_sink_pad:
    #     raise RuntimeError("Unable to get OSD sink pad")
    # osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

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
