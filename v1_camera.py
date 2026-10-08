import cv2 
import mediapipe as mp 
from mediapipe.tasks import python #Task API tool to configure models
from mediapipe.tasks.python import vision #land handmarker
import time 

#configuration - pretrain model:settings telling a program how to behave 
base_options = python.BaseOptions(
    model_asset_path = "hand_landmarker.task"
)

#configuration - MediaPipe hand dectection sys
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    running_mode=vision.RunningMode.VIDEO, # a sequence of video frames
    num_hands=2 
) 

landmarker = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break 

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRBG, data=rgb_frame)
    timestamps_ms = int(time.monotonic()*1000) #For MP to know frame orders


    cv2.imshow("ASL Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
landmarker.close()
cv2.destroyAllWindows()