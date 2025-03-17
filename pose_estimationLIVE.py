import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
# from ultralytics import YOLO # type: ignore
from pose_estimationV2 import Pose_Detection 


def main():
    print("This is the main function.")

    print("Starting webcam...")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # cap = cv.VideoCapture(r'Videos\Bicep_Curl2.mov')
    p_Time = 0 
    detector = Pose_Detection(upper_body_only = False)
    last_key = 'n'
    all_landmarks = []


    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Error: Failed to read frame or end of video reached.")
            break

        frame,success = detector.detect_person(frame,True)
        # landmarks = detector.detect_landmark(frame,success,ids = [1,2,3])
        landmarks = detector.detect_landmark(frame,success)
        # print(landmarks[0])
        # x = int(landmarks[14,1])
        # y = int(landmarks[14,2])
        # cv.circle(frame, (x, y), 15, (0, 0, 255), cv.FILLED)


        # print(landmarks)
        font = cv.FONT_HERSHEY_SIMPLEX
        c_Time = time.time()
        fps = 1/(c_Time-p_Time)
        p_Time =c_Time



        cv.putText(frame,  
                    str(int(fps)),  
                    (50, 50),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv.LINE_4)
        
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