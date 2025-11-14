#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSP → nvurisrcbin(HWデコード/動的pad) → queue → nvvideoconvert
    → capsfilter(NV12@NVMM) → nvstreammux → nvinfer(YOLOv8n検出)
    → nvvideoconvert → capsfilter(RGBA@NVMM) → nvdsosd(黒塗りprobe)
    → nvvideoconvert → nvegltransform → nveglglessink(ローカル表示)

- q キーで終了（GLib IO watch / 非同期1文字読み）
- pyds の BatchMeta 取得API差異に互換対応
- YOLOv8n：汎用物体検出モデル（80クラス対応）

# 基本的な使用方法 
python3 simple_rtsp_local_yolov8n.py "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101"

# 設定ファイルを明示的に指定 
python3 simple_rtsp_local_yolov8n.py "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" --infer-config ./config_infer_primary_face_yolov8n.txt

# 解像度とフレームレートを指定 
python3 simple_rtsp_local_yolov8n.py "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" --width 1920 --height 1080 --fps 30

# TCP接続を使用する場合（UDPで接続できない場合） 
python3 simple_rtsp_local_yolov8n.py "rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101" --tcp
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

LOG = logging.getLogger("rtsp-local-yolov8n")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30

# YOLOv8n COCOデータセットのクラスID（person）
# 顔検出ではなく人物全体を検出する場合
PERSON_CLASS_ID = 0

def parse_args():
    p = argparse.ArgumentParser(description="RTSP→YOLOv8n検出→物体黒塗り→ローカル表示（DeepStream）")
    p.add_argument("rtsp_url", help="rtsp://user:pass@ip:554/Streaming/Channels/101")
    p.add_argument("--infer-config", default="./config_infer_primary_face_yolov8n.txt",
                   help="nvinfer の設定ファイル（YOLOv8n検出）")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--latency", type=int, default=200, help="RTSP latency(ms)")
    p.add_argument("--tcp", action="store_true", help="RTSPをTCPで受信（既定はUDP）")
    p.add_argument("--target-class", type=int, default=0,
                   help="黒塗り対象のクラスID（デフォルト: 0=person）")
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

# --- DeepStream Pythonバインディング差異に対応した BatchMeta 取得 ---
def get_batch_meta(gst_buffer):
    """
    DSのバージョンで pyds API 名称が違うため互換取得を試みる。
    優先: batch_meta_from_buffer(gst_buffer)
    次点: gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    どちらも無ければ None
    """
    # 1) 新しめ
    try:
        return pyds.batch_meta_from_buffer(gst_buffer)
    except AttributeError:
        pass
    # 2) 従来
    try:
        return pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    except AttributeError:
        pass
    return None

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """nvdsosd の前で、検出された物体のBBoxを黒塗り処理"""
    target_class_id = u_data
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = get_batch_meta(gst_buffer)
    if not batch_meta:
        # バインディングの差異/未ロードでも処理継続（描画スキップ）
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

            # 次のノードを先に覚えておく（削除してもループを進められるように）
            next_obj = l_obj.next

            # ターゲットクラス以外はフレームから削除 → nvdsosd には渡さない
            if obj_meta.class_id != target_class_id:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                l_obj = next_obj
                continue

            # ===== ターゲットクラスのみここに来る =====
            LOG.debug(f"Object detected: class_id={obj_meta.class_id}, confidence={obj_meta.confidence:.2f}")
            
            rp = obj_meta.rect_params

            # 黒塗り処理（背景を黒で塗りつぶし）
            rp.has_bg_color = 1
            rp.bg_color.set(0.0, 0.0, 0.0, 1.0)  # RGBA 黒・不透明
            rp.border_width = 0  # 枠線なし
            rp.border_color.set(0.0, 0.0, 0.0, 0.0)

            l_obj = next_obj

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


# --- 動的パッド連結: nvurisrcbin の src → queue の sink を接続 ---
def on_src_pad_added(src, pad, target):
    if pad.get_direction() != Gst.PadDirection.SRC:
        return
    sinkpad = target.get_static_pad("sink")
    if sinkpad.is_linked():
        return
    ret = pad.link(sinkpad)
    if ret == Gst.PadLinkReturn.OK:
        LOG.info("nvurisrcbin → queue をリンク")
    else:
        LOG.error(f"nvurisrcbin → queue のリンク失敗: {ret}")

def build_pipeline(args: argparse.Namespace) -> Gst.Pipeline:
    Gst.init(None)
    pipe = Gst.Pipeline.new("rtsp-local-yolov8n")
    if not pipe:
        raise RuntimeError("Pipeline作成失敗")

    # 入力: nvurisrcbin（H.264/H.265自動対応・再接続あり）
    src = make(
        "nvurisrcbin", "src",
        uri=args.rtsp_url,
        latency=args.latency,
        drop_on_latency=True,
        rtsp_reconnect_interval_sec=5,
        rtsp_reconnect_attempts=-1,
        enable_udpsrc_buffer_size=True,
        udp_buffer_size=1<<20,   # 1MB
        cudadec_memtype=0        # NVMM
    )
    if args.tcp:
        try:
            src.set_property("rtsp_transport", 0)  # 0: TCP
        except Exception:
            pass

    # 動的パッド対策: 受け側に queue
    q0 = make("queue", "q0", max_size_buffers=0, max_size_bytes=0, max_size_time=0)

    # NVMMを明示（NV12）→ mux へ
    nvconv_pre = make("nvvideoconvert", "nvconv-pre")
    caps_pre = make("capsfilter", "caps-pre",
                    caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    mux = make("nvstreammux", "mux",
               batch_size=1, width=args.width, height=args.height,
               live_source=True, buffer_pool_size=8)

    # Primary Inference（YOLOv8n検出）
    pgie = make("nvinfer", "pgie",
                config_file_path=args.infer_config, unique_id=1)

    # OSD は RGBA を好むため、前段で RGBA へ変換
    nvconv_rgba = make("nvvideoconvert", "nvconv-rgba")
    caps_rgba = make("capsfilter", "caps-rgba",
                     caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    osd = make("nvdsosd", "osd", process_mode=0, display_text=0)

    # 表示系
    nvconv_post = make("nvvideoconvert", "nvconv-post")
    xform = make("nvegltransform", "xform")
    sink = make("nveglglessink", "sink", sync=False)

    for e in (src, q0, nvconv_pre, caps_pre, mux, pgie, nvconv_rgba, caps_rgba, osd, nvconv_post, xform, sink):
        pipe.add(e)

    # ★ 動的パッドでリンク
    src.connect("pad-added", on_src_pad_added, q0)

    # 静的リンク
    assert q0.link(nvconv_pre)
    assert nvconv_pre.link(caps_pre)

    # mux のリクエストシンク
    try:
        sinkpad_mux = mux.request_pad_simple("sink_0")  # DS6.x+
    except AttributeError:
        sinkpad_mux = mux.get_request_pad("sink_0")     # 旧API
    if not sinkpad_mux:
        raise RuntimeError("mux sink_0 リクエスト失敗")

    if caps_pre.get_static_pad("src").link(sinkpad_mux) != Gst.PadLinkReturn.OK:
        raise RuntimeError("caps-pre → mux.sink_0 リンク失敗")

    # mux→pgie→RGBA→osd→表示
    assert mux.link(pgie)
    assert pgie.link(nvconv_rgba)
    assert nvconv_rgba.link(caps_rgba)
    assert caps_rgba.link(osd)
    assert osd.link(nvconv_post)
    assert nvconv_post.link(xform)
    assert xform.link(sink)

    # 黒塗りを OSD の手前に差し込む（検出矩形を塗りつぶし）
    osd_sink_pad = osd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("nvdsosd sink pad 取得失敗")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, args.target_class)

    LOG.info("パイプライン構築完了（nvurisrcbin→YOLOv8n→黒塗り→ローカル表示）")
    return pipe

# --- q キーで終了させるための GLib IO watch ---
def install_quit_key(loop: GLib.MainLoop):
    """
    端末で 'q' を押したら loop.quit()。
    GLib.IOChannel で1文字ノンブロッキング読み取り。
    """
    io = GLib.IOChannel.unix_new(sys.stdin.fileno())
    io.set_encoding(None)  # バイナリ扱い
    io.set_buffered(False)

    def _on_key(iochan, cond, user_data):
        try:
            status, data = iochan.read(1)  # 1文字読む
        except Exception:
            return True
        if status == GLib.IOStatus.NORMAL and data:
            ch = data.decode(errors="ignore").lower() if isinstance(data, (bytes, bytearray)) else data.lower()
            if ch == 'q':
                LOG.info("q キー検出 → 終了します")
                user_data.quit()
                return False  # これ以上　ウォッチしない
        return True

    GLib.io_add_watch(io, GLib.IOCondition.IN, _on_key, loop)

def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    # モデルディレクトリの存在確認と作成
    import os
    import glob
    import configparser
    
    model_dir = "./models/yolov8n"
    if not os.path.exists(model_dir):
        LOG.info(f"モデルディレクトリを作成: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
    
    # 設定ファイルのパス情報をログ出力
    LOG.info(f"設定ファイル: {args.infer_config}")
    try:
        config = configparser.ConfigParser()
        config.read(args.infer_config)
        if 'property' in config:
            onnx_file = config['property'].get('onnx-file', 'N/A')
            engine_file = config['property'].get('model-engine-file', 'N/A')
            
            # 絶対パスに変換して表示
            onnx_abs = os.path.abspath(onnx_file) if onnx_file != 'N/A' else 'N/A'
            engine_abs = os.path.abspath(engine_file) if engine_file != 'N/A' else 'N/A'
            
            LOG.info(f"設定: onnx-file = {onnx_file}")
            LOG.info(f"  → 絶対パス: {onnx_abs}")
            LOG.info(f"設定: model-engine-file = {engine_file}")
            LOG.info(f"  → 絶対パス: {engine_abs}")
    except Exception as e:
        LOG.warning(f"設定ファイルの読み込みに失敗: {e}")
    
    # 既存のengineファイルを確認
    engine_files = glob.glob(f"{model_dir}/*.engine")
    if engine_files:
        LOG.info(f"既存のTensorRTエンジンファイル: {', '.join(engine_files)}")
    else:
        LOG.info(f"TensorRTエンジンファイルが見つかりません。初回実行時に自動生成されます（数分かかります）")

    LOG.info(f"黒塗り対象クラスID: {args.target_class}")

    pipe = build_pipeline(args)
    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop, pipe)

    # Ctrl+C/SIGTERM でも終了
    def handle_sig(signum, frame):
        LOG.info(f"Signal {signum} を受信、停止")
        loop.quit()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # q キーで終了
    install_quit_key(loop)

    try:
        LOG.info("YOLOv8n 物体検出を開始")
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
