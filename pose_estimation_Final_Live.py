import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
# from ultralytics import YOLO # type: ignore
from pose_estimationV2 import Pose_Detection 


def countReps(angles,count,direction):
     # This will be our range when doing the bicep curl
     interperlation = np.interp(angles,(46,165),(0,100))
     bar =  np.interp(angles,(40,175),(-6,110))
    #  direction = 0 
    #  print('angle',angles)
    #  print('inter',interperlation)
    #  print('direction',direction)
     if interperlation == 100:
          if direction == 0:
               count+=.5
               direction = 1
     if interperlation == 0:
          if direction == 1:
               count+=.5
               direction = 0
    
     return count,direction,bar

def main():
    print("This is the main function.")

    print("Starting webcam...")
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1520)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1000)

    # cap = cv.VideoCapture(r'Videos\Bicep_Curl2.mov')
    p_Time = 0 
    detector = Pose_Detection(upper_body_only = False)
    last_key = 'n'
    all_landmarks = []
    Bicep_L_reps = 0 
    Bicep_L_direct = 0
    Bicep_R_reps = 0 
    Bicep_R_direct = 0
    Leg_L_reps = 0 
    Leg_L_direct = 0
    Leg_R_reps = 0 
    Leg_R_direct = 0
    L_bar = 0 
    R_bar = 0 


    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Error: Failed to read frame or end of video reached.")
            break

        frame,success = detector.detect_person(frame,False)
        landmarks = detector.detect_landmark(frame,success,False)
        # landmarks = detector.detect_landmark(frame,success,ids = [1,2,3])

        # IF I put this all in a method would the method need to be apart of the object
        # landmarks = detector.detect_landmark(frame,success,False)
        # right_arm_lm = detector.detect_landmark(frame,success,draw=False,ids = [12,14,16])
        # left_arm_lm = detector.detect_landmark(frame,success,draw=False, ids = [11,13,15])
        # Bicep_L_reps, Bicep_L_direct, Bicep_R_reps, Bicep_R_direct = detector.Bicep_counter(landmarks,frame,success,
        #                                                                                     Bicep_L_reps, Bicep_L_direct, Bicep_R_reps, Bicep_R_direct)


        if last_key == 'b':
            # landmarks = detector.detect_landmark(frame,success,False)
            # right_arm_lm = detector.detect_landmark(frame,success,draw=False,ids = [12,14,16])
            # left_arm_lm = detector.detect_landmark(frame,success,draw=False, ids = [11,13,15])
            Bicep_L_reps, Bicep_L_direct, Bicep_R_reps, Bicep_R_direct = detector.Bicep_counter(frame,success,
                                                                                            Bicep_L_reps, Bicep_L_direct,
                                                                                            Bicep_R_reps, Bicep_R_direct)
        
        elif last_key == 'i':
            # landmarks = detector.detect_landmark(frame,success,False)
            # right_arm_lm = detector.detect_landmark(frame,success,draw=False,ids = [12,14,16])
            # left_arm_lm = detector.detect_landmark(frame,success,draw=False, ids = [11,13,15])
            Bicep_L_reps, Bicep_L_direct, Bicep_R_reps, Bicep_R_direct = detector.Bench_counter(frame,success,
                                                                                            Bicep_L_reps, Bicep_L_direct,
                                                                                            Bicep_R_reps, Bicep_R_direct)
        
        elif last_key == 's':
            Leg_L_reps, Leg_L_direct, Leg_R_reps, Leg_R_direct = detector.Squats_counter(frame,success,
                                                                                            Leg_L_reps, Leg_L_direct,
                                                                                            Leg_R_reps, Leg_R_direct,L_bar,R_bar)
        
             
        elif last_key == 'n':
            Bicep_L_reps = 0 
            Bicep_L_direct = 0
            Bicep_R_reps = 0 
            Bicep_R_direct = 0
            Leg_L_reps = 0 
            Leg_L_direct = 0
            Leg_R_reps = 0 
            Leg_R_direct = 0

        
             
        all_landmarks.append(landmarks)

        # Display the resulting frame 
        cv.imshow('Webcame', frame) 


        k = cv.waitKey(1) 
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
