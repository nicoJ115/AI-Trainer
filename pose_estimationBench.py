import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
# from ultralytics import YOLO # type: ignore
from pose_estimationV2 import Pose_Detection 
from scipy import ndimage as nd



def sharpen(img):
    # print("sharpen")
    # sharpened_image =[]
    img = img/255
    kernel = np.array([[ -1/9, -1/9,  -1/9],
                        [-1/9,  18/9, -1/9],
                        [ -1/9, -1/9,  -1/9]])



        # Apply the convolution to sharpen the image
    sharpened = np.zeros_like(img)
    for i in range(3):
        
        # sharpened = np.zeros_like(img)
        sharpened[:, :, i] = nd.convolve(img[:, :,i], kernel)
    sharpened = sharpened *255
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return np.array(sharpened)
    

def countReps(landmarks,angles,count,direction):
     # This will be our range when doing the bicep curl
     interperlation = np.interp(angles,(50,130),(0,100))
     bar =  np.interp(angles,(35,175),(-15,145))
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
     
    #  print('count',count)
     return count,direction,bar 

def main():
    print("This is the main function.")

    print("Starting webcam...")
    # cap = cv.VideoCapture(0)
    cap = cv.VideoCapture(r'Videos\Gold_Standard\Gold_Incline_Bench3.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # cap = cv.VideoCapture(r'Videos\Bicep_Curl2.mov')
    p_Time = 0 
    detector = Pose_Detection(upper_body_only = False)
    last_key = 'n'
    all_landmarks = []
    L_reps = 0 
    L_direct = 0
    R_reps = 0 
    R_direct = 0
    reps = 0
    angleRange = 0
    font = cv.FONT_HERSHEY_SIMPLEX
    counter = 0


    while True:
        success, frame = cap.read()
        # frame = cv.resize(frame,(720,720))
        if not success or frame is None:
            print("Error: Failed to read frame or end of video reached.")
            break
        # frame = cv.resize(frame, (720, 720))
        # frame = sharpen(frame)
        frame,success = detector.detect_person(frame,False)
        # landmarks = detector.detect_landmark(frame,success,ids = [1,2,3])
        landmarks = detector.detect_landmark(frame,success,False)
        '''
        Instead of giving left and right reps for bench what I could do get the reps of the one it actually counting 
        # After that just make sure the bar is straight
        '''
        right_arm_lm = detector.detect_landmark(frame,success,draw=False,ids = [12,14,16])
        left_arm_lm = detector.detect_landmark(frame,success,draw=False, ids = [11,13,15])
        # if len(left_arm_lm)==3:
        #     angle,tanAngle = detector.findAngle(frame,left_arm_lm[0],left_arm_lm[1],left_arm_lm[2])
        #     L_reps,L_direct,L_bar = countReps(left_arm_lm,angle,L_reps,L_direct) 
        #     Left_arm_reps = 'Left arms reps: '+ str(int(L_reps))
            
        #     Left_perect = 'range of motion, '+str(int(L_bar)) + '%'
        #     cv.putText(frame,
        #             Left_arm_reps, 
        #             (25, 25),  
        #             font,
        #             .5,  
        #             (0, 255, 255),  
        #             2,  
        #             cv.LINE_4)
        
        #     cv.putText(frame,
        #                 Left_perect, 
        #                 (25, 50),  
        #                 font,
        #                 .50,  
        #                 (0, 255, 255),  
        #                 2,  
        #                 cv.LINE_4)


        # if len(right_arm_lm)==3:
        #     Rangle,RtanAngle = detector.findAngle(frame,right_arm_lm[0],right_arm_lm[1],right_arm_lm[2])
        #     R_reps,R_direct,R_bar = countReps(right_arm_lm,Rangle,R_reps,R_direct) 
        #     Right_arm_reps = 'Right arms reps: '+ str(int(R_reps))
        #     Right_perect = 'range of motion, '+str(int(R_bar)) + '%'
        #     (text_width, text_height), _ = cv.getTextSize(Right_arm_reps, font, .5, 2)

        #     frame_height, frame_width = frame.shape[:2]
        #     x = frame_width - text_width - 25  # 25 px padding from right edge
        #     y = 25

        #     cv.putText(frame,
        #             Right_arm_reps, 
        #             (x, y),  
        #             font,
        #             .5,  
        #             (0, 255, 255),  
        #             2,  
        #             cv.LINE_4)
        
        #     cv.putText(frame,
        #                 Right_perect, 
        #                 (x, y+25),  
        #                 font,
        #                 .50,  
        #                 (0, 255, 255),  
        #                 2,  
        #                 cv.LINE_4)

        if len(left_arm_lm)==3 and len(right_arm_lm)==3:
            Langle,LtanAngle = detector.findAngle(frame,left_arm_lm[0],left_arm_lm[1],left_arm_lm[2],True)
            L_reps,L_direct,L_bar = countReps(left_arm_lm,Langle,L_reps,L_direct) 
            Left_arm_reps = 'Left arms reps: '+ str(int(L_reps))
            

            Rangle,RtanAngle = detector.findAngle(frame,right_arm_lm[0],right_arm_lm[1],right_arm_lm[2],True)
            R_reps,R_direct,R_bar = countReps(right_arm_lm,Rangle,R_reps,R_direct) 
            Right_arm_reps = 'Right arms reps: '+ str(int(R_reps))
            


            # Assuming each is (x, y) or [x, y]
            left_wrist_y = left_arm_lm[2][1]
            right_wrist_y = right_arm_lm[2][1]

            # Tolerance value to allow for slight offset (in pixels)
            tolerance = 20
            # (255,0,0) = Blue 
            # (255,0,0) = Green 
            # (0,0,255) = Red
            color = (0,0,0)
            # print(abs(left_wrist_y - right_wrist_y))
            if abs(left_wrist_y - right_wrist_y) <= 15:
                bar_status = "Bar is straight"
                #  (255,0,0)
                color = (0,255,0)
            else:
                bar_status = "Bar is tilted"
                #  (255,0,0)
                color = (0,0,255)

 
            if Left_arm_reps>=Right_arm_reps:
                reps=L_reps
                angleRange = L_bar
            else:
                reps=R_reps
                angleRange = R_bar 

            Arm_reps = 'Bench Reps: '+ str(int(reps))
            Arm_perect = 'range of motion, '+str(int(angleRange)) + '%'
            

            cv.putText(frame,
                    Arm_reps, 
                    (25, 25),  
                    font,
                    .5,  
                    (0, 255, 255),  
                    2,  
                    cv.LINE_4)
        
            cv.putText(frame,
                        Arm_perect, 
                        (50, 50),  
                        font,
                        .50,  
                        (0, 255, 255),  
                        2,  
                        cv.LINE_4)
            
            cv.putText(frame,
                        bar_status, 
                        (100, 100),  
                        font,
                        .50,  
                        color,  
                        2,  
                        cv.LINE_4)


        # print(landmarks)
        font = cv.FONT_HERSHEY_SIMPLEX
        c_Time = time.time()
        fps = 1/(c_Time-p_Time)
        p_Time =c_Time

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
