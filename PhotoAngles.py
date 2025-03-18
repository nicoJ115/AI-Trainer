import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
from pose_estimationV2 import Pose_Detection as pm 
# from ultralytics import YOLO # type: ignore


#print code 
def main():
    print("This is the main function.")

    # cap = cv.VideoCapture(r'Videos\Bicep_Curl2.mov')
    # img = cv.imread(r'Photos\Bicep_Angle2.png')
    # img = cv.resize(img,(580,720))
    # cv.imshow("Img",img)
    # cv.waitKey(0) 
    detector = pm()
    last_key = 'n'
    counter  = 0
    while True:
        img = cv.imread(r'Photos\Bicep_Angle3.png')
        img = cv.resize(img,(580,720))
        img,suc = detector.detect_person(img,draw=False)
        right_arm_lm = detector.detect_landmark(img,suc,draw=False,ids = [12,14,16])
        left_arm_lm = detector.detect_landmark(img,suc,draw=False, ids = [11,13,15])
        detector.findAngle(img,left_arm_lm[0],left_arm_lm[1],left_arm_lm[2])
        # counter  = 0 
        if counter == 0:
            print(right_arm_lm)
            print(left_arm_lm)
            counter=1
        if img is None:
            print("Error: Failed to read frame or end of video reached.")
            break

        cv.imshow("Img",img)
        k = cv.waitKey(1) 
        if k>=0:
                last_key = chr(k)

        if last_key == 'q':
                break


if __name__ == "__main__":
    main()