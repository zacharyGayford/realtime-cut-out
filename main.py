import numpy as np
import matplotlib.pyplot as plt

import os

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

# import fast sam
from fastsam import FastSAM, FastSAMPrompt

import cv2

# plotting functions
def plot_image(img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img / 255.0)

def plot_overlay(overlay):
    plt.imshow(overlay / 255.0, alpha = 0.5)

# yolo model
yolo_model = YOLO("yolov8n.pt")

def yolo_get_bounding_boxes(img):
    result = yolo_model.predict(source=img)
    result = result[0].boxes.xyxy.tolist()
    if len(result) == 0:
        return None
    return np.array(result[0])[None, :]

fast_sam_model = FastSAM("fastsam.pt")

def fast_sam_get_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box = yolo_get_bounding_boxes(img)
    if box is None:
        return None
    masks = fast_sam_model(img, device="cpu", conf=0.4)
    prompt = FastSAMPrompt(img, masks, device="cpu")
    return prompt.box_prompt(bboxes=box.tolist())

# cut out image
def cut_out_image(img):
    background = np.ones_like(img) * 255
    background = background.astype(np.uint8)
    mask = fast_sam_get_mask(img)
    if mask is None:
        return background
    mask = np.where(mask > 0.5, 1, 0)
    mask = np.rollaxis(mask, 0, 3)
    print(mask.shape)
    new_image = background * (1 - mask) + img * (mask)
    return new_image.astype(np.uint8)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("cannot open video capture")
    exit()

while(True):
    retval, frame = video.read()
    print(frame.shape)
    frame = cut_out_image(frame)
    print(frame.shape)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
       break

video.release()
cv2.destroyAllWindows()
