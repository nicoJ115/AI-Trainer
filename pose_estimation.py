import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
# from ultralytics import YOLO # type: ignore


cap = cv.VideoCapture(r'Videos\2.mov')
#This imports the Pose module from MediaPipe’s solutions package.
#It provides the necessary tools for detecting and tracking human body poses
mpPose = mp.solutions.pose
#This imports drawing utilities from MediaPipe.
#These utilities help visualize the detected pose by drawing keypoints and connections on an image/frame.
mpDraw = mp.solutions.drawing_utils
'''
This initializes the Pose model.
It detects and tracks 33 key points of the human body (such as shoulders, elbows, knees, etc.).
By default, it uses a pre-trained deep learning model to perform real-time pose detection.
'''
pose = mpPose.Pose()
# img_in = cv.resize(np.float32(img_in)/255, (1280, 720))
last_key = 'n'
while True:
    success, frame = cap.read()
    # frame =  cv.resize(np.float32(frame), (1280, 720))
    if not success or frame is None:
        print("Error: Failed to read frame or end of video reached.")
        break

    # Convert the image from BGR to RGB since the cv color are inverted
    frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    # Process the frame with the pose model
    result = pose.process(frameRGB)
    # print(result.pose_landmarks)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame,result.pose_landmarks,mpPose.POSE_CONNECTIONS)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame,  
                'TEXT ON VIDEO',  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv.LINE_4)
    # Display the resulting frame 
    cv.imshow('video', frame) 
    # cv.waitKey(1)
    k = cv.waitKey(1) 
    if k>=0:
            prev_key = last_key
            last_key = chr(k)

    if last_key == 'q':
            break

    # creating 'q' as the quit  
    # button for the video 
    # if cv.waitKey(1) & 0xFF == ord('q'): 
    #     break
    # cv.waitKey(1)
    


# Test OpenCV
print("OpenCV Version:", cv.__version__)

# Test MediaPipe
print("MediaPipe is working!")

# Test YOLO
# model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano model
print("YOLOv8 is not working!")

# release the cap object 
cap.release() 
# close all windows 
cv.destroyAllWindows()