import cv2 
import mediapipe as mp 
from mediapipe.tasks import python #Task API tool to configure models
from mediapipe.tasks.python import vision #land handmarker
import time 
from mediapipe.tasks.python.vision import HandLandmarksConnections
import numpy as np 
from pathlib import Path #create folders/file path
from datetime import datetime #gives recording unique filename 

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

previous_time = time.perf_counter()
sequence = [] 

key_to_label = {
    ord('a') : "A",
    ord('j') : "J", 
    ord('n') : "NAME"
}

data_dir = Path("v1_data")
data_dir.mkdir(exist_ok = True)

recording_label = None
countdown_start = None
is_recording = False
recording_start = None
saved_message = None 

while True:
    success, frame = cap.read()

    if not success:
        break 

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.monotonic()*1000) #For MP to know frame orders
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    left_features = [0.0] * 66
    right_features = [0.0] * 66 
    
    #Cicles (landmark display) for both hand 
    if result.hand_landmarks:
        height, width, _ = frame.shape

        for hand in result.hand_landmarks:
            #circle
            for landmark in hand: 
                x = int(landmark.x * width)#OpenCV needs pixel coordinates
                y = int(landmark.y * height)#normilized -> pixel coord
                cv2.circle(frame, (x,y), 4, (0,0,255), -1)
            
            for connection in HandLandmarksConnections.HAND_CONNECTIONS: #skeleton lines
                start = hand[connection.start] #.start attribute prebuild
                end = hand[connection.end]

                start_point = (int(start.x * width), int(start.y * height))
                end_point = (int(end.x * width), int(end.y * height))
                cv2.line(frame, start_point, end_point, (255,0,0), 2)
        
        #Extract features for each detected hand
        for i, handLandmark in enumerate(result.hand_landmarks):
            wrist = handLandmark[0]
            features = []
        
            #Relative Coordinates
            for landmark in handLandmark: 
                features.extend([#extend add each ele individually
                    landmark.x - wrist.x, #relative mark vs wrist
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ])
            features.extend([wrist.x, wrist.y]) #Describe the path hand follows 
            features.append(1.0)#hand presence 
            #66 features = 63 relative landmark + 2 absolute wrist position + 1 presence
            hand_label = result.handedness[i][0].category_name

            if hand_label == "Left":
                left_features = features
            else:
                right_features = features

    frame_features = left_features + right_features
    
    if is_recording:
        sequence.append(frame_features) 
        if len(sequence) == 30:
            is_recording = False

            recording_duration = time.perf_counter() - recording_start
            print(f"Recording duration: {recording_duration:.2f} seconds")

            sign_dir = data_dir / recording_label #v1_data/J
            sign_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = sign_dir / f"{timestamp}.npy"

            np.save(file_path, np.array(sequence, dtype=np.float32))
            print("Saved:", file_path)
            saved_message = f"Saved {recording_label}! Press A/J/N to record again."
            print("Sequence shape:", len(sequence), len(sequence[0]))
            recording_label = None
        
    current_time = time.perf_counter()
    fps = 1/ (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    if countdown_start is not None:
        elapsed = time.perf_counter() - countdown_start
        remaining = max(0, 3 - elapsed)
        cv2.putText(
            frame, f"Get ready: {remaining:.1f}s",
            (20,90),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2
        )
        if elapsed >= 3:
            sequence.clear()
            countdown_start = None
            is_recording = True
            recording_start = time.perf_counter()
            print("Recording started for:", recording_label)
        
    if is_recording:
        cv2.putText(
            frame, 
            f"RECORDING: {recording_label} ({len(sequence)}/30)", 
            (20,130), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, #font size
            (0,0,0), 
            2 #text thickness
        )
    
    if saved_message is not None:
        cv2.putText(
            frame, 
            saved_message,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0,255,0),
            2
        )

    cv2.imshow("ASL Camera", frame) #imageshow(window title, frame cap from camera)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break 
    
    if key in key_to_label and recording_label is None:
        saved_message = None
        recording_label = key_to_label[key]
        countdown_start = time.perf_counter()

cap.release()
landmarker.close()
cv2.destroyAllWindows()

