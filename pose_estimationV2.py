import numpy as np
import cv2 as cv # type: ignore
import time
import mediapipe as mp # type: ignore
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
        landmarks = []
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
                        landmarks.append([id,cx,cy, lm.visibility])
                elif self.upper_body_only:
                    if id < 25:  # Landmark IDs 0 to 22 generally represent the upper body
                        # print("Why",id)
                        landmarks.append([id,cx,cy, lm.visibility])
                else:
                    # print("Why",id)
                    landmarks.append([id,cx,cy, lm.visibility])
                if draw:
                    # print('')
                    # cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
                    pass
            return np.array(landmarks)
        return np.array(landmarks)
        



def main():
    print("This is the main function.")

    cap = cv.VideoCapture(r'Videos\Dead_lift1.mov')
    p_Time = 0 
    detector = Pose_Detection(upper_body_only = False)
    # print(detector.upper_body_only)
    # print()
    last_key = 'n'
    all_landmarks = []
    while True:
        success, frame = cap.read()
        frame,success = detector.detect_person(frame,True)
        if not success or frame is None:
            print("Error: Failed to read frame or end of video reached.")
            break
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
        cv.imshow('video', frame) 
        cv.waitKey(1)
        k = cv.waitKey(1) 
        if k>=0:
                prev_key = last_key
                last_key = chr(k)

        if last_key == 'q':
                break


    # print(landmarks)

if __name__ == "__main__":
    main()