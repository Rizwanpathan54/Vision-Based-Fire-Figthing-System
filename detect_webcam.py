import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import serial
import time
import threading
import torch


#CONFIG 
CAM_L = 10
CAM_R = 6

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15 

BASELINE_M = 0.140
FX_PIXELS  = 580.0
SMOOTH_WINDOW = 5

CAMERA_TO_PANTILT_X = 0.075   
CAMERA_TO_PANTILT_Y = 0.0     

# SERIAL CONFIG 
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("Serial Connected.")
except Exception as e:
    print(f"Serial Error: {e}. Running in simulation mode.")
    ser = None

def send_serial_async(msg):
    """Sends serial data in a background thread to prevent camera timeouts."""
    if ser and ser.is_open:
        def write_task():
            try:
                # Convert string to bytes if necessary
                data = msg.encode() if isinstance(msg, str) else msg
                ser.write(data)
                ser.flush() # Ensure data is sent
            except Exception as e:
                print(f"Serial Write Error: {e}")
        
        threading.Thread(target=write_task, daemon=True).start()

# Object Detector MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"!!! SYSTEM STATUS: Running YOLO on {device.upper()} !!!")
model = YOLO("best.pt").to(device)

# THREADED CAMERA CLASS
class CameraStream:
    def __init__(self, idx):
        self.idx = idx
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                # grab() clears the buffer, retrieve() gets the latest
                if self.cap.grab():
                    ret, frame = self.cap.retrieve()
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
            time.sleep(0.01) # Keep thread from maxing out CPU

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()

#Initialize Streams
streamL = CameraStream(CAM_L)
streamR = CameraStream(CAM_R)

#LOGIC HELPERS
def get_flame_center(result_obj):
    """Modified to take a single result object from a batch list."""
    if result_obj.boxes is None or len(result_obj.boxes) == 0:
        return None, None, None

    best_area = 0
    best_center = None
    best_bbox = None

    for box in result_obj.boxes:
        if int(box.cls[0]) != 0: continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            best_bbox = (int(x1), int(y1), int(x2), int(y2))

    return (best_center[0], best_center[1], best_bbox) if best_center else (None, None, None)

# GUI SETUP
root = tk.Tk()
root.title(f"Stereo Flame Tracker - {device.upper()} Accelerated")

labelL = tk.Label(root); labelL.grid(row=0, column=0)
labelR = tk.Label(root); labelR.grid(row=0, column=1)
info_label = tk.Label(root, text="Initializing...", font=("Arial", 14))
info_label.grid(row=1, column=0, columnspan=2)

depth_hist = deque(maxlen=SMOOTH_WINDOW)
flame_locked = False

# MAIN LOOP
def update():
    global flame_locked

    okL, frameL = streamL.read()
    okR, frameR = streamR.read()

    if not okL or not okR or frameL is None or frameR is None:
        root.after(10, update)
        return

    # 1.BATCH INFERENCE - Runs both frames in one GPU pass
    results = model([frameL, frameR], conf=0.6, verbose=False, device=device)

    # 2.Extract Centers
    cxL, cyL, bboxL = get_flame_center(results[0])
    cxR, cyR, bboxR = get_flame_center(results[1])

    flame_detected = (cxL is not None and cxR is not None)

    if flame_detected:
        disparity = abs(cxL - cxR)
        if disparity > 2:
            z = (FX_PIXELS * BASELINE_M) / disparity
            depth_hist.append(z)
        
        if depth_hist:
            avg_z = np.mean(depth_hist)
            pan_cam = np.arctan((cxL - FRAME_WIDTH/2) / FX_PIXELS)
            tilt_cam = np.arctan((cyL - FRAME_HEIGHT/2) / FX_PIXELS)
            
            pan = np.degrees(np.arctan((avg_z * np.tan(pan_cam) - CAMERA_TO_PANTILT_X) / avg_z))
            tilt = np.degrees(np.arctan((avg_z * np.tan(tilt_cam) - CAMERA_TO_PANTILT_Y) / avg_z))

            # 3.ASYNC SERIAL - Never blocks the camera loop
            if not flame_locked:
                msg = f"PAN:{pan:.2f},TILT:{tilt:.2f},DEPTH:{avg_z:.2f}\n"
                send_serial_async(msg)
                flame_locked = True
            send_serial_async(b"FIRE:1\n")
            
            info_label.config(text=f"🔥 FIRE (GPU): {avg_z:.2f}m", fg="red")
    else:
        if flame_locked:
            send_serial_async(b"FIRE:0\n")
            flame_locked = False
        depth_hist.clear()
        info_label.config(text="Scanning...", fg="black")

    #4.Drawing
    if bboxL: cv2.rectangle(frameL, (bboxL[0], bboxL[1]), (bboxL[2], bboxL[3]), (0,0,255), 2)
    if bboxR: cv2.rectangle(frameR, (bboxR[0], bboxR[1]), (bboxR[2], bboxR[3]), (0,0,255), 2)

    #5.GUI Refresh
    imgL = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)))
    imgR = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)))
    labelL.configure(image=imgL); labelL.image = imgL
    labelR.configure(image=imgR); labelR.image = imgR

    root.after(10, update) # Fast polling to keep USB buffers clear

def on_closing():
    streamL.stop()
    streamR.stop()
    if ser: ser.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
update()
root.mainloop()