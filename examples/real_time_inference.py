# This script was written by Claude Opus 5.5.
"""Real-time character detection overlay for Roblox (Windows, NVIDIA GPU, TensorRT engines only).

Captures the Roblox window, runs a TensorRT engine of a YOLO model on the GPU (resize, inference and NMS all on the GPU),
tracks the characters with ByteTrack and draws their boxes on a transparent, click-through overlay on top of the game.
The script loads a TensorRT .engine file only; it does not run .pt or .pth files, and it does not support Giraffe.

1. Build the engine on the GPU that will run it, from the repository root (input shape matches the model's 16:9 input):
       yolo export model=models/rblx-yolo-cheetah.pt format=engine quantize=fp16 batch=1 imgsz=576,1024
2. Start Roblox, then run:
       python examples/real_time_inference.py
3. Stop with Ctrl+C in the console.

To use another model, export it the same way (Rabbit: imgsz=736,1280, Orca: imgsz=864,1536) and change ENGINE below.
"""
import ctypes
import threading
import time
from ctypes import wintypes
import pygame
import win32api
import win32con
import psutil
import win32gui
import win32process
import struct
import pathlib
import numpy as np, tensorrt as trt, torch, torch.nn.functional as F
from types import SimpleNamespace
from torchvision.ops import nms
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "models" / "rblx-yolo-cheetah.engine"
if not ENGINE.exists():
    raise SystemExit(f"{ENGINE} not found. Build it first (see the top of this file or the README's Export section).")

ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))

DWMWA_EXTENDED_FRAME_BOUNDS = 9

def get_program_geometry(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    window = {
        "x": rect.left, "y": rect.top,
        "w": rect.right - rect.left, "h": rect.bottom - rect.top,
    }

    # Client area (game content only) in desktop coordinates
    _, _, cw, ch = win32gui.GetClientRect(hwnd)
    cx, cy = win32gui.ClientToScreen(hwnd, (0, 0))
    client = {"x": cx, "y": cy, "w": cw, "h": ch}

    # Monitor the window is on, and whether it fills it
    monitor = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
    ml, mt, mr, mb = win32api.GetMonitorInfo(monitor)["Monitor"]
    fullscreen = (cx, cy, cx + cw, cy + ch) == (ml, mt, mr, mb)

    return window, client, fullscreen

EXE_NAME = "RobloxPlayerBeta.exe"

def find_program_hwnd():
    program_pids = {
        p.pid for p in psutil.process_iter(["name"])
        if p.info["name"] and p.info["name"].lower() == EXE_NAME.lower()
    }
    if not program_pids:
        raise RuntimeError("Program is not running")

    matches = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or not win32gui.GetWindowText(hwnd):
            return True
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid in program_pids:
            matches.append(hwnd)
        return True

    win32gui.EnumWindows(callback, None)
    if not matches:
        raise RuntimeError("Program process found, but no visible window")
    return matches[0]

hwnd = find_program_hwnd()
print(f"Capturing: {win32gui.GetWindowText(hwnd)} (hwnd={hwnd})")

capture = WindowsCapture(
    cursor_capture=False,
    draw_border=False,
    window_hwnd=hwnd,
)

latest_frame = None
new_frame = threading.Event()

@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    global latest_frame
    latest_frame = frame.frame_buffer.copy()   # BGRA as delivered; detect() flips channels on the GPU
    new_frame.set()

@capture.event
def on_closed():
    print("Capture session closed")

control = capture.start_free_threaded()

monitor_width = ctypes.windll.user32.GetSystemMetrics(0)
monitor_height = ctypes.windll.user32.GetSystemMetrics(1)

pygame.init()

window = pygame.Window("Overlay", (monitor_width, monitor_height), borderless=True, always_on_top=True, resizable=True)
screen = window.get_surface()
screen.set_alpha(255)

font = pygame.font.SysFont("Arial", 15)

pyg_hwnd = window.handle

win32gui.SetWindowPos(pyg_hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
win32gui.SetWindowLong(pyg_hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(
    pyg_hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
win32gui.SetLayeredWindowAttributes(pyg_hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)

running = True

rects = []          # boxes to draw; replaced as a whole by the vision thread (atomic)
MAX_VISION_FPS = 200

CONF, IOU = 0.3, 0.7

def detect(image, _s={}):    # image: HxWx3 BGR or HxWx4 BGRA uint8, 16:9 -> (N,6): x1 y1 x2 y2 track_id conf

    if not _s:   # first call only: load the engine, bind GPU buffers, create the tracker
        raw = ENGINE.read_bytes()
        eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(raw[4 + struct.unpack("<i", raw[:4])[0]:])
        global engH, engW
        _, _, engH, engW = eng.get_tensor_shape("images")
        print(engW, engH)
        _s["ctx"] = eng.create_execution_context()
        _s["x"], _s["o"] = (torch.zeros(tuple(eng.get_tensor_shape(n)), device="cuda") for n in ("images", "output0"))
        _s["ctx"].set_tensor_address("images", _s["x"].data_ptr())
        _s["ctx"].set_tensor_address("output0", _s["o"].data_ptr())
        _s["trk"] = BYTETracker(SimpleNamespace(track_high_thresh=CONF, track_low_thresh=0.1, new_track_thresh=CONF,
                                                track_buffer=60, match_thresh=0.8, fuse_score=True))

    h, w = image.shape[:2]
    x = torch.from_numpy(image).cuda(non_blocking=True)[..., [2, 1, 0]].permute(2, 0, 1)[None].float().div_(255)
    _s["x"].copy_(F.interpolate(x, size=(engH, engW), mode="bilinear", antialias=False))     # GPU resize
    _s["ctx"].execute_async_v3(torch.cuda.current_stream().cuda_stream)
    o = _s["o"][0]; k = o[4] > 0.1          # keep low scores: ByteTrack uses them to hold tracks
    cx, cy, bw, bh, sc = o[0][k], o[1][k], o[2][k], o[3][k], o[4][k]
    b = torch.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], 1)
    keep = nms(b, sc, IOU)[:300]            # GPU NMS
    s = torch.tensor([w / engW, h / engH], device="cuda").repeat(2)
    d = torch.cat([b[keep] * s, sc[keep, None], torch.zeros_like(sc[keep, None])], 1).cpu().numpy()
    t = _s["trk"].update(Boxes(d, (h, w)))  # rows: x1 y1 x2 y2 id conf cls idx
    return t[:, :6].astype(np.float32) if len(t) else np.zeros((0, 6), np.float32)


def yolo_track():
    global rects, running
    last_print, n = time.time(), 0
    min_gap = 1.0 / MAX_VISION_FPS
    while running:
        if not new_frame.wait(0.1):          # sleep until the capture thread delivers a NEW frame
            continue
        new_frame.clear()
        t0 = time.time()
        results = detect(latest_frame)
        rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, tid, conf in results if conf >= 0.7]   # empty list clears old boxes
        n += 1
        if time.time() - last_print > 2:
            print(f"Vision FPS: {n / (time.time() - last_print):.0f}")
            last_print, n = time.time(), 0
        spare = min_gap - (time.time() - t0)
        if spare > 0:
            time.sleep(spare)                # leave GPU time for the game


threading.Thread(target=yolo_track, daemon=True).start()

try:
    clock = pygame.time.Clock()
    last_geo, last_print, n = 0.0, time.time(), 0
    pos = size = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if time.time() - last_geo > 0.25:    # window geometry rarely changes; no need to query it every frame
            last_geo = time.time()
            program_window, client, fullscreen = get_program_geometry(hwnd)
            if pos != (program_window["x"], program_window["y"]):
                pos = (program_window["x"], program_window["y"])
                window.position = pos
            if size != (program_window["w"], program_window["h"]):
                size = (program_window["w"], program_window["h"])
                window.size = size

        screen = window.get_surface()
        if screen:
            screen.fill((0, 0, 0, 255))
            for x1, y1, x2, y2 in rects:
                pygame.draw.rect(screen, (255, 10, 10), (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 3)
        window.flip()

        n += 1
        if time.time() - last_print > 2:
            print(f"Overlay FPS: {n / (time.time() - last_print):.0f}")
            last_print, n = time.time(), 0
        clock.tick(60)
finally:
    running = False
    pygame.quit()
    control.stop()
