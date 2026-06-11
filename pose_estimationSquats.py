import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
# from ultralytics import YOLO # type: ignore
from pose_estimationV2 import Pose_Detection 
import panda as pd
import os
import datetime


def countReps(angles,back_angle,count,direction):
     # This will be our range when doing the bicep curl
    interperlation = np.interp(angles,(45,150),(0,100))
    bar =  np.interp(angles,(40,175),(-5,125))
    back_angle = abs(back_angle)

    if interperlation == 100 and back_angle<=180 and back_angle>=160:
            if direction == 0:
                count+=.5
                direction = 1
    elif interperlation == 0 and back_angle<=80 and back_angle>=60:
            if direction == 1:
                count+=.5
                direction = 0
    #  print('count',count)
    return count,direction,bar 

def main():
    print("This is the main function.")

    print("Starting webcam...")
    # cap = cv.VideoCapture(0)
    cap = cv.VideoCapture(r'Videos\Gold_Standard\Gold_Squats1.mp4')
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # cap = cv.VideoCapture(r'Videos\Bicep_Curl2.mov')
    p_Time = 0 
    detector = Pose_Detection(upper_body_only = False)
    last_key = 'n'
    # all_landmarks = []
    L_reps = 0 
    L_direct = 0
    R_reps = 0 
    R_direct = 0
    L_bar = 0
    R_bar = 0


    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Error: Failed to read frame or end of video reached.")
            break
        # frame = frame[:,::-1]
        frame,success = detector.detect_person(frame,False)
        # landmarks = detector.detect_landmark(frame,success,ids = [1,2,3])
        landmarks = detector.detect_landmark(frame,success,False)
        right_leg_lm = detector.detect_landmark(frame,success,draw=False,ids = [24,26,28])
        left_leg_lm = detector.detect_landmark(frame,success,draw=False, ids = [23,25,27])
        left_back_lm = detector.detect_landmark(frame,success,draw=False, ids = [26,24,12])
        right_back_lm = detector.detect_landmark(frame,success,draw=False, ids = [25,23,11])

        L_straight = 0
        R_straight = 0

        if len(left_leg_lm)==3 and len(left_back_lm) == 3:
            angle,tanAngle =  detector.findAngle(frame,left_leg_lm[0],left_leg_lm[1],left_leg_lm[2])
            back_angle,back_tanAngle = detector.findAngle(frame,left_back_lm[0],left_back_lm[1],left_back_lm[2],draw= False,drawLongAngle=False)
            L_reps,L_direct,L_bar = countReps(angle,back_tanAngle,L_reps,L_direct)
            L_straight = back_tanAngle

        if len(right_leg_lm)==3 and len(right_back_lm) == 3:
            angle,tanAngle = detector.findAngle(frame,right_leg_lm[0],right_leg_lm[1],right_leg_lm[2])
            back_angle,back_tanAngle = detector.findAngle(frame,right_back_lm[0],right_back_lm[1],right_back_lm[2],draw= False,drawLongAngle=True)
            R_reps,R_direct,R_bar = countReps(angle,back_tanAngle,R_reps,R_direct)
            R_straight = back_tanAngle

        font = cv.FONT_HERSHEY_SIMPLEX

        
        Left_leg_reps = 'Left leg reps: '+ str(int(L_reps))
        Right_leg_reps = 'Right leg reps: '+ str(int(R_reps))

        Left_perect = 'range of motion, '+str(int(L_bar)) + '%'
        Right_perect = 'range of motion, '+str(int(R_bar)) + '%'

        (text_width, text_height), _ = cv.getTextSize(Right_leg_reps, font, .5, 2)

        frame_height, frame_width = frame.shape[:2]
        x = frame_width - text_width - 25  # 25 px padding from right edge
        y = 25
        reps = 0
        angleRange = 0
        side = ''
        Back_straightness = 0
        color = (0,255,0)
        if L_reps>=R_reps:
                reps=L_reps
                angleRange = L_bar
                side = 'L'
                Back_straightness = L_straight
        else:
                reps=R_reps
                angleRange = R_bar
                side = 'R' 
                Back_straightness = R_straight


        straight_back = ''
        
        if Back_straightness<=180 and Back_straightness>=160 or Back_straightness<=80 and Back_straightness>=60:
                  straight_back = 'Your back is not straight'
                  color = (0,0,255)
        else: 
                  straight_back = 'Your back is  straight'
                  color = (0,255,0)
        
                  
        Leg_reps = 'Squats Reps: '+ str(int(reps))
        Leg_perect = 'range of motion, '+str(int(angleRange)) + '%'





        cv.putText(frame,
                    Leg_reps, 
                    (25, 25),  
                    font,
                    .5,  
                    (0, 255, 255),  
                    2,  
                    cv.LINE_4)
        
        cv.putText(frame,
                        Leg_perect, 
                        (50, 50),  
                        font,
                        .50,  
                        (0, 255, 255),  
                        2,  
                        cv.LINE_4)
            
        cv.putText(frame,
                        straight_back, 
                        (100, 100),  
                        font,
                        .50,  
                        color,  
                        2,  
                        cv.LINE_4)
        

        # Display the resulting frame 
        cv.imshow('Webcame', frame) 


        k = cv.waitKey(30) 
        if k>=0:
                prev_key = last_key
                last_key = chr(k)

        if last_key == 'q':
                break

    cap.release()
    cv.destroyAllWindows()

    # print(landmarks)

if __name__ == "__main__":
    main()
