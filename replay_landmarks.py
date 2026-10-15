import numpy as np 
from pathlib import Path 
import cv2

#Find all label folders
data_dir = Path("v1_data")

labels = sorted([
    folder for folder in data_dir.iterdir() #Paths
    if folder.is_dir()
])
if not labels: 
    raise SystemExit(f"No label folder found.")

#Display available labels
for i, label in enumerate(labels):
    print(f"{i}: {label.name}")

label_choice = int(input("Which label? (Index)"))
selected_label = labels[label_choice]
print("Selected label path:", selected_label)

#Find record in selected label
files = sorted(selected_label.glob("*.npy"))
if not files:
    raise SystemExit(f"No recordings found for {selected_label.name}.")
print(f"\nRecordings for {selected_label.name}:")

for i, file in enumerate(files):
    print(f"{i}:{file.name}")

#Select recording
file_choice = int(input("Which recording? (Index)"))
file_path = files[file_choice]
sequence = np.load(file_path)
#load and return numpy arr from file

print("File:", file_path)
print("Shape:", sequence.shape)

def draw_hand(canvas, landmarks, wrist_x, wrist_y):
    for landmark in landmarks:
        relative_x = landmark[0]
        relative_y = landmark[1]

        abs_x = relative_x + wrist_x
        abs_y = relative_y + wrist_y

        pixel_x = int(abs_x * width)
        pixel_y = int(abs_y * height)

        cv2.circle(canvas, (pixel_x, pixel_y), 8, (0,255,0), -1)

    #Color code hand connections
    connections = []

    #Thumb (Blue)
    for i in range(1,4):
        connections.append((i, i+1, (255,0,0)))

    #Index, Middle, Ring Fingers (White)
    for start in [5,9,13]:
        for i in range(start, start + 3):
            connections.append((i,i+1, (255,255,255)))
    
    #Pinky (Red)
    for i in range(17,20):
        connections.append((i, i+1, (0,0,255)))
    
    #Connect wrist to base of each finger 
    for base in [1, 5, 9, 13, 17]:
        if base ==1:
            color = (255,0,0)
        elif base == 17:
            color = (0,0,255)
        else:
            color = (255,255,255)
        connections.append((0,base,color))
    
    #Connect knuckles across palm
    for start in [5,9,13]:
        connections.append((start, start+4, (128,0,128)))

    for start, end, color in connections:
        start_x = int((landmarks[start][0] + wrist_x) * width)
        start_y = int((landmarks[start][1] + wrist_y) * height)

        end_x = int((landmarks[end][0] + wrist_x) * width)
        end_y = int((landmarks[end][1] + wrist_y) * height)

        cv2.line(canvas, (start_x, start_y), (end_x, end_y), color, 2)

#Convert normalized coordinates -> pixel coordinates 
#Draw Hand 
width = 640 
height = 480
frame_index = 0 #Start at first frame

while True: 
    #1. Get current select frame 
    frame_features = sequence[frame_index]

    #2. Extract both hands 
    # Left hand
    left_features = frame_features[:66]
    left_landmarks = left_features[:63].reshape(21, 3)
    l_wrist_x = left_features[63]
    l_wrist_y = left_features[64]
    left_present = left_features[65]

    # Right hand
    right_features = frame_features[66:]
    right_landmarks = right_features[:63].reshape(21, 3)
    r_wrist_x = right_features[63]
    r_wrist_y = right_features[64]
    right_present = right_features[65]

    #3. Create new black canvas 
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    #4. Draw code (hand dots + color connections)
    if left_present == 1.0:
        draw_hand(canvas, left_landmarks, l_wrist_x, l_wrist_y)
    if right_present == 1.0:
        draw_hand(canvas, right_landmarks, r_wrist_x, r_wrist_y)
    
    #5. Display selected frame 
    cv2.putText(
        canvas,
        f"Frame {frame_index+1}/{len(sequence)}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )
    cv2.imshow("ASL Landmark Replay", canvas)
    
    #6 Wait for arrow keys
    key = cv2.waitKeyEx(0)

    if key == 63235: #Right arrow 
        frame_index = min(frame_index + 1, len(sequence) - 1)
    elif key == 63234: #Left arrow
        frame_index = max(frame_index-1, 0)
    elif key == ord('q'):
        break

    print("Current frame:", frame_index)

cv2.destroyAllWindows()