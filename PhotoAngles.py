import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
from pose_estimationV2 import Pose_Detection as pm 
# from ultralytics import YOLO # type: ignore



def main():
    print("This is the main function.")

    # cap = cv.VideoCapture(r'Videos\Bicep_Curl2.mov')
    # img = cv.imread(r'Photos\Bicep_Angle2.png')
    # img = cv.resize(img,(580,720))
    # cv.imshow("Img",img)
    # cv.waitKey(0) 
    detector = pm()
    last_key = 'n'
    while True:
        img = cv.imread(r'Photos\Bicep_Angle3.png')
        img = cv.resize(img,(580,720))
        img,suc = detector.detect_person(img)
        landmarks = detector.detect_landmark(img,suc,ids = [12,14,16])
        # print(img)
        if img is None:
            print("Error: Failed to read frame or end of video reached.")
            break

        cv.imshow("Img",img)
        k = cv.waitKey(1) 
        if k>=0:
                last_key = chr(k)

        if last_key == 'q':
                break

#     p_Time = 0 
#     detector = Pose_Detection(upper_body_only = False)
#     print(detector.upper_body_only)
#     print()
#     last_key = 'n'
#     all_landmarks = []
#     while True:
#         success, frame = cap.read()
#         frame,success = detector.detect_person(frame,True)
#         if not success or frame is None:
#             print("Error: Failed to read frame or end of video reached.")
#             break
#         # landmarks = detector.detect_landmark(frame,success,ids = [1,2,3])
#         landmarks = detector.detect_landmark(frame,success,False)
#         print(landmarks[0])
#         # x = int(landmarks[14,1])
#         # y = int(landmarks[14,2])
#         # cv.circle(frame, (x, y), 15, (0, 0, 255), cv.FILLED)


#         # print(landmarks)
#         font = cv.FONT_HERSHEY_SIMPLEX
#         c_Time = time.time()
#         fps = 1/(c_Time-p_Time)
#         p_Time =c_Time



#         cv.putText(frame,  
#                     str(int(fps)),  
#                     (50, 50),  
#                     font, 1,  
#                     (0, 255, 255),  
#                     2,  
#                     cv.LINE_4)
        
#         all_landmarks.append(landmarks)

#         # Display the resulting frame 
#         cv.imshow('video', frame) 
        
#         k = cv.waitKey(1) 
#         if k>=0:
#                 last_key = chr(k)

#         if last_key == 'q':
#                 break


#     # print(landmarks)

if __name__ == "__main__":
    main()