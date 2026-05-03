import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import math
import serial
import time
import os
from datetime import datetime

#CONFIGURATION
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
FPS              = 30

YOLO_MODEL_PATH  = "best.pt"
CONF_THRESH      = 0.6
DEPTH_ROI_RADIUS = 3

SERIAL_PORT      = "/dev/ttyUSB0"
BAUD_RATE        = 115200
SEND_INTERVAL    = 0.08          # seconds between serial writes
RECORD_OUTPUT_DIR = "recordings"

#CAMERA → PAN-TILT TRANSFORM (0, 5, -6 cm)
tx, ty, tz = 0.0, 0.05, -0.06

T_cam_to_pt = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0,  1]
], dtype=np.float32)

#SERIAL SETUP

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("[INFO] Serial connected")
except Exception:
    ser = None
    print("[WARN] Serial not connected (simulation mode)")

last_send_time = 0.0
flame_locked   = False

# DEVICE + YOLO MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] YOLO running on {device.upper()}")

model = YOLO(YOLO_MODEL_PATH).to(device)

fp16_enabled = False

#REALSENSE SETUP
pipeline = rs.pipeline()
config   = rs.config()

config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16,  FPS)

profile    = pipeline.start(config)
align      = rs.align(rs.stream.color)

intrinsics = (
    profile.get_stream(rs.stream.color)
           .as_video_stream_profile()
           .get_intrinsics()
)

depth_scale = (
    profile.get_device()
           .first_depth_sensor()
           .get_depth_scale()
)

print(f"[INFO] RealSense started | depth scale = {depth_scale:.4f} m/unit")

#VIDEO RECORDING SETUP
os.makedirs(RECORD_OUTPUT_DIR, exist_ok=True)
recording_path = os.path.join(
    RECORD_OUTPUT_DIR,
    f"flame_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    recording_path,
    fourcc,
    FPS,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

if video_writer.isOpened():
    print(f"[INFO] Recording enabled -> {recording_path}")
else:
    print(f"[WARN] Could not open video writer -> {recording_path}")
    video_writer = None

#MAIN LOOP
try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

        result = model(color_image, conf=CONF_THRESH, verbose=False)[0]

        if not fp16_enabled and device == "cuda":
            model.half()
            fp16_enabled = True
            print("[INFO] FP16 (half-precision) enabled after warmup")

        flame_detected = False

        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) != 0:
                    continue

                flame_detected = True

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                r   = DEPTH_ROI_RADIUS
                roi = depth_image[
                    max(0, cy - r) : min(FRAME_HEIGHT, cy + r + 1),
                    max(0, cx - r) : min(FRAME_WIDTH,  cx + r + 1)
                ]
                valid = roi[roi > 0]

                if valid.size == 0:
                    continue

                Z_cam = float(np.mean(valid))   # depth from CAMERA (metres)

                #3D POINT (CAMERA FRAME)
                Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                    intrinsics, [cx, cy], Z_cam
                )

                #MATRIX TRANSFORMATION
                p_cam = np.array([Xc, Yc, Zc, 1.0], dtype=np.float32)
                p_pt  = T_cam_to_pt @ p_cam

                Xp, Yp, Zp = p_pt[:3]   # position in pan-tilt frame


                #PAN / TILT ANGLES
                pan  = math.degrees(math.atan2(Xp, Zp))
                tilt = math.degrees(math.atan2(Yp, Zp))

                #SERIAL SEND
                now = time.time()
                if ser and (now - last_send_time) > SEND_INTERVAL:
                    msg = f"PAN:{pan:.2f},TILT:{tilt:.2f},DEPTH:{Zp:.2f},FIRE:1\n"
                    ser.write(msg.encode())
                    last_send_time = now
                    flame_locked   = True

                #DEBUG PRINT
                print(
                    f"[FLAME] "
                    f"Depth_cam={Z_cam:.3f} m | "
                    f"Depth_pt={Zp:.3f} m  | "
                    f"Pan={pan:.2f}°  | Tilt={tilt:.2f}°"
                )
                # VISUALISATION
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    color_image,
                    f"CamZ:{Z_cam:.2f}m  PTZ:{Zp:.2f}m",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
                cv2.putText(
                    color_image,
                    f"P:{pan:.1f}  T:{tilt:.1f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                break   #track only the first (highest-conf) flame box

        if video_writer:
            cv2.putText(
                color_image,
                "REC",
                (FRAME_WIDTH - 80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            cv2.circle(color_image, (FRAME_WIDTH - 100, 24), 8, (0, 0, 255), -1)
            video_writer.write(color_image)

        # FLAME LOST
        if not flame_detected and flame_locked:
            if ser:
                ser.write(b"FIRE:0\n")
            flame_locked = False

        cv2.imshow("Flame Tracking (Camera vs PanTilt Depth)", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#CLEANUP
finally:
    pipeline.stop()
    if video_writer:
        video_writer.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
    print("[INFO] Shutdown complete")
