import threading
import dxcam
import os
import cv2
import time
import numpy as np
import win32gui
import win32con
import keyboard
import pygame
from ultralytics import YOLO

running = True

def on_press(e):
    global running
    if str(e.name)=="t":
        running=False
keyboard.on_press(on_press)

def cv2_to_pygame(frame):
    return pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR")

cam = dxcam.create(output_color="BGR")
cam.start(target_fps=240)

pygame.init()
screen = pygame.display.set_mode((640, 480), pygame.NOFRAME)
screen.set_alpha(128)

hwnd = pygame.display.get_wm_info()["window"]

model = YOLO("../models/rblx-yolo-cheetah.pt", task="detect")

dt = 0
lt = time.time()

img = np.zeros((1, 1, 3), np.uint8)

frame_queue = []

def yolo_detect():
    global running,dt,lt,img
    while running:

        fresh_img = cam.get_latest_frame()
        if fresh_img is not None:
            lt = time.time()
            results = model.predict(fresh_img, verbose=False)
            now = time.time()
            dt = now - lt
    
            detections = results[0].boxes
    
            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)
    
                classidx = int(detections[i].cls.item())
    
                conf = detections[i].conf.item()
    
                cv2.rectangle(fresh_img, (xmin, ymin), (xmax, ymax), (0, int(255 * conf), int(255 * conf)), 4)
                cv2.putText(fresh_img, "conf: " + str(int(conf * 100)), (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255 if classidx == 0 else 0, 255, 255), 2)

        img = fresh_img
    cam.stop()
    cam.release()

threading.Thread(target=yolo_detect).start()

while running:
    if dt != 0:
        print("fps: "+str(1/dt))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if img is not None:
        pyg_img = cv2_to_pygame(cv2.resize(img,(640,480)))
        screen.blit(pyg_img,(0,0))
    pygame.display.update()

pygame.quit()
keyboard.unhook_all()
