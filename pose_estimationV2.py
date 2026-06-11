import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
import math
# from ultralytics import YOLO # type: ignore

class Pose_Detection():

    """
        Initializes the Pose_Detection object with custom settings.

        Parameters:
        - mode (bool): If True, processes each frame as a static image (no tracking). Default is False.
        - complexity (int): Model complexity (0, 1, 2) for balancing speed vs. accuracy. Default is 1.
        - smooth_landmarks (bool): If True, smooths landmark positions to reduce jitter. Default is True.
        - enable_segmentation (bool): If True, enables background segmentation. Default is False.
        - smooth_segmentation (bool): If True, smooths segmentation masks. Default is True.
        - min_detection_confidence (float): Minimum confidence for detecting poses (0 to 1). Default is 0.5.
        - min_tracking_confidence (float): Minimum confidence for tracking poses (0 to 1). Default is 0.5.
        - upper_body_only (bool): If True, detects only upper body (available in older versions of MediaPipe). Default is False.
    """
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, upper_body_only=False):

        # Initialize MediaPipe Pose and Drawing utilities
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.landmarks = []

        # Create a Pose object with the provided parameters
        self.pose = self.mpPose.Pose(
            static_image_mode=mode,
            model_complexity=complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Store the upper body only setting
        self.upper_body_only = upper_body_only
    
    def detect_person(self,frame,draw = True):
        frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        result = self.pose.process(frameRGB)
        # print(result.pose_landmarks)
        if draw:
             if result.pose_landmarks:
                # self.mpDraw.draw_landmarks(frame,result.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                # print(result.pose_landmarks)
                self.mpDraw.draw_landmarks(
                                            frame,
                                            result.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS,
                                            self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green dots for landmarks
                                            self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=3)  # Red lines for connections
                                        )


        return frame, result

    def detect_landmark(self,img,result,draw = True,ids = []):
        # print("hi")
        self.landmarks = []
        if result.pose_landmarks:
            for id, lm in enumerate(result.pose_landmarks.landmark):
                # print('image shape',img.shape)
                # print(id,lm)
                row, col, color = img.shape
                # Filter upper body landmarks if upper_body_only is True
                cx, cy = int(lm.x*col),int(lm.y*row)

                # landmarks.append([id,cx,cy, lm.visibility])
                if ids:
                    # print("why")
                    if id in ids:
                        self.landmarks.append([id,cx,cy, lm.visibility])
                        if draw:
                            cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED) 
                elif self.upper_body_only:
                    if id < 25:  # Landmark IDs 0 to 22 generally represent the upper body
                        # print("Why",id)
                        self.landmarks.append([id,cx,cy, lm.visibility])
                        if draw:
                            cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED) 
                else:
                    # print("Why",id)
                    self.landmarks.append([id,cx,cy, lm.visibility])
                    if draw:
                            cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED) 
                # if draw:
                #     print('')
                #     cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
                #     pass
            return np.array(self.landmarks)
        return np.array(self.landmarks)
    
    # Now need to get the calculation for this problem 
    # I used this site to get the formaula for 3 points
    # https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
    def findAngle(self,img,pos1,pos2,pos3,draw = True,drawLongAngle=False):
         pos1_x, pos1_y,_ = pos1[1:]
         pos1_x,pos1_y = int (pos1_x), int(pos1_y)
         pos2_x, pos2_y,_ = pos2[1:]
         pos2_x,pos2_y = int (pos2_x), int(pos2_y)
         pos3_x, pos3_y,_ = pos3[1:]
         pos3_x,pos3_y = int (pos3_x), int(pos3_y)

         A = np.array([pos1_x - pos2_x, pos1_y - pos2_y])
         B = np.array([pos3_x - pos2_x, pos3_y - pos2_y])

         # Dot product and magnitudes
         dot_product = np.dot(A, B)
         magnitude_A = np.linalg.norm(A)
         magnitude_B = np.linalg.norm(B)
         magnitude_A = math.sqrt(A[0]**2 + A[1]**2)
         magnitude_B = math.sqrt(B[0]**2 + B[1]**2)
         # Compute angle in radians and convert to degrees
         try:
             angle = math.degrees(math.acos(dot_product / (magnitude_A * magnitude_B)))
         except ValueError as e:
             print("Caught an error:", e)
             angle = 90

         angle1 = math.atan2(pos1_y - pos2_y, pos1_x - pos2_x)  # Angle of vector pos2 → pos1
         angle2 = math.atan2(pos3_y - pos2_y, pos3_x - pos2_x)  # Angle of vector pos2 → pos3

         # Compute angle difference
         tanAngle = math.degrees(angle2 - angle1)

         
        #  tanAngle = abs(tanAngle)
        #  if tanAngle > 180:
        #     tanAngle = 360 - tanAngle
        #  if tanAngle > 180:
        #     tanAngle -= 360
        #  elif tanAngle < -180:
        #     tanAngle += 360


         if draw:
              cv.line(img, (pos1_x,pos1_y),(pos2_x,pos2_y),(0,255,0),2)
              cv.line(img, (pos3_x,pos3_y),(pos2_x,pos2_y),(0,255,0),2)
              cv.circle(img, (pos1_x, pos1_y), 10, (0, 0, 255), 2)
              cv.circle(img, (pos1_x, pos1_y), 5, (0, 0, 255), cv.FILLED)
              cv.circle(img, (pos2_x, pos2_y), 10, (0, 0, 255), 2)
              cv.circle(img, (pos2_x, pos2_y), 5, (0, 0, 255), cv.FILLED)
              cv.circle(img, (pos3_x, pos3_y), 10, (0, 0, 255), 2)
              cv.circle(img, (pos3_x, pos3_y), 5, (0, 0, 255), cv.FILLED)
              # WE dont need to print the tan angles we just need to secure the facts
              #  that if the tan angles is greater than 180
              # In other words if the person is over extending.
              if drawLongAngle:
                cv.putText(img, str(int(tanAngle)), (pos2_x - 50, pos2_y - 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
              else: 
                cv.putText(img, str(int(angle)), (pos2_x - 40, pos2_y - 20), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)
         return angle,tanAngle

    def countRepsBiceps(self,angles,angle2,count,direction):
        # This will be our range when doing the bicep curl
        interperlation = np.interp(angles,(46,165),(0,100))
        bar =  np.interp(angles,(40,175),(-6,110))
        if (100 <= angle2 <= 120) or (65 <= angle2 <= 95):
            if interperlation == 100:
                if direction == 0:
                    count+=.5
                    direction = 1
            if interperlation == 0:
                if direction == 1:
                    count+=.5
                    direction = 0
        
        return count,direction,bar     

    def countRepsBench(self,angles,count,direction):
        # This will be our range when doing the bicep curl
        interperlation = np.interp(angles,(50,130),(0,100))
        bar =  np.interp(angles,(35,175),(-15,145))
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

    def countRepsSquats(self,angles,back_angle,count,direction):
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
        
    def Bicep_counter(self, frame,success,L_reps, L_direct, R_reps, R_direct):
        right_arm_lm = self.detect_landmark(frame,success,draw=False,ids = [12,14,16])
        left_arm_lm = self.detect_landmark(frame,success,draw=False, ids = [11,13,15])

        right_sholder = self.detect_landmark(frame,success,draw=False,ids = [11,12,14])
        left_sholder = self.detect_landmark(frame,success,draw=False,ids = [13,11,12])

        L_bar = 0 
        L_straight = 0
        R_straight = 0
        R_bar = 0
        if len(left_arm_lm)==3 and len(left_sholder)==3:
            angle,tanAngle =self.findAngle(frame,left_arm_lm[0],left_arm_lm[1],left_arm_lm[2],True)
            sangle,stanAngle = self.findAngle(frame,left_sholder[1],left_sholder[0],left_sholder[2],False)
            L_reps,L_direct,L_bar = self.countRepsBiceps(angle,sangle,L_reps,L_direct)
            L_straight = sangle

        if len(right_arm_lm)==3 and len(right_sholder)==3:
            # print("hi")
            angle,tanAngle = self.findAngle(frame,right_arm_lm[0],right_arm_lm[1],right_arm_lm[2],True)
            sangle,stanAngle = self.findAngle(frame,right_sholder[0],right_sholder[1],right_sholder[2],False)
            R_reps,R_direct,R_bar = self.countRepsBiceps(angle,sangle,R_reps,R_direct) 
            R_straight = sangle
        

             

        Left_arm_reps = 'Left arms reps: '+ str(int(L_reps))
        Right_arm_reps = 'Right arms reps: '+ str(int(R_reps))

        Left_perect = 'range of motion, '+str(int(L_bar)) + '%'
        Right_perect = 'range of motion, '+str(int(R_bar)) + '%'

        L_arm_straight = ''
        L_color = (0,0,0)
        if (100 <= L_straight <= 120) or (65 <= L_straight <= 95):
            L_arm_straight = 'Left arm is straight'
            L_color = (0,255,0)
        else: 
            L_arm_straight = 'Left arm is not straight: '+ str(int(L_straight))
            L_color = (0,0,255)

        R_arm_straight = ''
        R_color = (0,0,0)
        if (100 <= R_straight <= 120) or (65 <= R_straight <= 95):
            R_arm_straight = 'Right arm is straight'
            R_color = (0,255,0)
        else: 
            R_arm_straight = 'Right arm is not straight: '+ str(int(R_straight))
            R_color = (0,0,255)

        font = cv.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), _ = cv.getTextSize(Right_arm_reps, font, .5, 2)

        frame_height, frame_width = frame.shape[:2]
        x = frame_width - text_width - 25  # 25 px padding from right edge
        y = 25

        cv.putText(frame,
                    Left_arm_reps, 
                    (25, 25),  
                    font,
                    .5,  
                    (0, 0, 0),  
                    2,  
                    cv.LINE_4)
        
        cv.putText(frame,
                    Left_perect, 
                    (25, 50),  
                    font,
                    .50,  
                    (0, 0, 0),  
                    2,  
                    cv.LINE_4)
        
        cv.putText(frame,
                    L_arm_straight, 
                    (25, 75),  
                    font,
                    .50,  
                    L_color,  
                    2,  
                    cv.LINE_4)
        
        cv.putText(frame,
                    Right_arm_reps, 
                    (x, y),  
                    font,
                    .5,  
                    (0, 0, 0),  
                    2,  
                    cv.LINE_4)
        
        cv.putText(frame,
                    Right_perect, 
                    (x, y+25),  
                    font,
                    .50,  
                    (0, 0, 0),  
                    2,  
                    cv.LINE_4)
        
        cv.putText(frame,
                    R_arm_straight, 
                    (x-50, y+50),  
                    font,
                    .50,  
                    R_color,  
                    2,  
                    cv.LINE_4)
        
        return L_reps, L_direct, R_reps, R_direct
        # pass


    def Bench_counter(self, frame,success,L_reps, L_direct, R_reps, R_direct):
        right_arm_lm = self.detect_landmark(frame,success,draw=False,ids = [12,14,16])
        left_arm_lm = self.detect_landmark(frame,success,draw=False, ids = [11,13,15])
        font = cv.FONT_HERSHEY_SIMPLEX
        # L_reps,L_direct,L_bar = 0,0,0
        # R_reps,R_direct,R_bar = 0,0,0
        if len(left_arm_lm)==3 and len(right_arm_lm)==3:
            angle,tanAngle = self.findAngle(frame,left_arm_lm[0],left_arm_lm[1],left_arm_lm[2],False)
            # R_reps,R_direct,R_bar = self.countReps(angle,R_reps,R_direct) 
            L_reps,L_direct,L_bar = self.countRepsBench(angle,L_reps,L_direct) 
            Left_arm_reps = 'Left arms reps: '+ str(int(L_reps))
            

            angle,tanAngle = self.findAngle(frame,right_arm_lm[0],right_arm_lm[1],right_arm_lm[2],False)
            R_reps,R_direct,R_bar = self.countRepsBench(angle,R_reps,R_direct) 
            Right_arm_reps = 'Right arms reps: '+ str(int(R_reps))
            


            # Assuming each is (x, y) or [x, y]
            left_wrist_y = left_arm_lm[2][2]
            # print(left_arm_lm[2])
            right_wrist_y = right_arm_lm[2][2]

            # Tolerance value to allow for slight offset (in pixels)
            tolerance = 20
            # (255,0,0) = Blue 
            # (255,0,0) = Green 
            # (0,0,255) = Red
            color = (0,0,0)
            # print(abs(left_wrist_y - right_wrist_y))
            if abs(left_wrist_y - right_wrist_y) <= tolerance:
                bar_status = "Bar is straight"
                #  (255,0,0)
                color = (0,255,0)
            else:
                bar_status = "Bar is tilted: "+str(abs(left_wrist_y - right_wrist_y))
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
            
            return L_reps, L_direct, R_reps, R_direct

        return L_reps, L_direct, R_reps, R_direct
    
    def Squats_counter(self, frame,success,L_reps, L_direct, R_reps, R_direct,L_bar,R_bar):
        right_leg_lm = self.detect_landmark(frame,success,draw=False,ids = [24,26,28])
        left_leg_lm = self.detect_landmark(frame,success,draw=False, ids = [23,25,27])
        left_back_lm = self.detect_landmark(frame,success,draw=False, ids = [26,24,12])
        right_back_lm = self.detect_landmark(frame,success,draw=False, ids = [25,23,11])
        L_straight = 0
        R_straight = 0

        if len(left_leg_lm)==3 and len(left_back_lm) == 3:
            angle,tanAngle =  self.findAngle(frame,left_leg_lm[0],left_leg_lm[1],left_leg_lm[2])
            back_angle,back_tanAngle = self.findAngle(frame,left_back_lm[0],left_back_lm[1],left_back_lm[2],draw= True,drawLongAngle=True)
            L_reps,L_direct,L_bar = self.countRepsSquats(angle,back_tanAngle,L_reps,L_direct)
            L_straight = back_tanAngle

        if len(right_leg_lm)==3 and len(right_back_lm) == 3:
            angle,tanAngle = self.findAngle(frame,right_leg_lm[0],right_leg_lm[1],right_leg_lm[2])
            back_angle,back_tanAngle = self.findAngle(frame,right_back_lm[0],right_back_lm[1],right_back_lm[2],draw= False,drawLongAngle=True)
            R_reps,R_direct,R_bar = self.countRepsSquats(angle,back_tanAngle,R_reps,R_direct)
            R_straight = back_tanAngle

        font = cv.FONT_HERSHEY_SIMPLEX

        
        Left_leg_reps = 'Left leg reps: '+ str(int(L_reps))
        Right_leg_reps = 'Right leg reps: '+ str(int(R_reps))


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
        
        if (Back_straightness<=180 and Back_straightness>=160) or (Back_straightness<=80 and Back_straightness>=60):
                  straight_back = 'Your back is straight: '+ str(int(Back_straightness))
                  color = (0,255,0)
        else:
                straight_back = 'Your back is overextended or underextended: '+ str(int(Back_straightness))
                color = (0,0,255)
        
                  
        Leg_reps = 'Squats Reps: '+ str(int(reps))
        Leg_perect = 'range of motion, '+str(int(angleRange)) + '%'

        cv.putText(frame,
                    Leg_reps, 
                    (25, 25),  
                    font,
                    .5,  
                    (0, 0, 0),  
                    2,  
                    cv.LINE_4)
        
        cv.putText(frame,
                        Leg_perect, 
                        (50, 50),  
                        font,
                        .50,  
                        (0, 0, 0),  
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
        
        return L_reps, L_direct, R_reps, R_direct



    
def main():
    print("This is the main function.")

    cap = cv.VideoCapture(r'Videos\MVI_3475.mov')
    p_Time = 0 
    detector = Pose_Detection(upper_body_only = False)
    print(detector.upper_body_only)
    print()
    last_key = 'n'
    all_landmarks = []
    while True:
        success, frame = cap.read()
        frame,success = detector.detect_person(frame,True)
        if not success or frame is None:
            print("Error: Failed to read frame or end of video reached.")
            break
        # landmarks = detector.detect_landmark(frame,success,ids = [1,2,3])
        landmarks = detector.detect_landmark(frame,success,True)
        # landmarks = detector.detect_landmark(frame,success)
        right_arm_lm = detector.detect_landmark(frame,success,draw=False,ids = [24,26,28])
        left_arm_lm = detector.detect_landmark(frame,success,draw=False, ids = [23,25,29])
        if len(left_arm_lm)==3:
            detector.findAngle(frame,left_arm_lm[0],left_arm_lm[1],left_arm_lm[2])
        if len(right_arm_lm)==3:
            detector.findAngle(frame,right_arm_lm[0],right_arm_lm[1],right_arm_lm[2])
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
        cv.imshow('video', frame) 
        
        k = cv.waitKey(1) 
        if k>=0:
                last_key = chr(k)

        if last_key == 'q':
                break


    # print(landmarks)

if __name__ == "__main__":
    main()
