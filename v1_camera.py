import cv2 
import mediapipe as mp 
from mediapipe.tasks import python #Task API tool to configure models
from mediapipe.tasks.python import vision #land handmarker
import time 
from mediapipe.tasks.python.vision import HandLandmarksConnections

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
while True:
    success, frame = cap.read()

    if not success:
        break 

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.monotonic()*1000) #For MP to know frame orders
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    print("Hands detected:", len(result.hand_landmarks), flush=True)

    left_features = [0.0] * 66
    right_features = [0.0] * 66 
    
    #Cicles (landmark display) for both hand 
    if result.hand_landmarks:
        hand_label = result.handedness[0][0].category_name
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
            print("Hand label:", hand_label)

            if hand_label == "Left":
                left_features = features
            else:
                right_features = features

    frame_features = left_features + right_features
    print("Frame features:", len(frame_features))
    print("Left:", len(left_features), left_features[-1])
    print("Right:", len(right_features), right_features[-1])

    current_time = time.perf_counter()
    fps = 1/ (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("ASL Camera", frame) #imageshow(window title, frame cap from camera)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
landmarker.close()
cv2.destroyAllWindows()