import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp
from ultralytics import YOLO\


    

# Test OpenCV
print("OpenCV Version:", cv.__version__)

# Test MediaPipe
print("MediaPipe is working!")

# Test YOLO
model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano model
print("YOLOv8 is working!")
