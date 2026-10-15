# ASL Sign Recognition System 🖐

An American Sign Language (ASL) recognition project using MediaPipe hand landmarks and Python. Currently developing a sequence based recognition pipeline focused on data collection, validation, and movement recognition.

## Tech Stack

**Python · MediaPipe · OpenCV · NumPy · PyTorch (LSTM planned)**

## Current Implementation

- Real time webcam hand tracking using MediaPipe
- Extracts 21 landmarks per hand with wrist relative coordinates
- Supports left and right hands with 132 features per frame
- Uses absolute wrist positions and hand presence flags
- Records 30 frame sequences as NumPy (`.npy`) files
- Interactive landmark replay viewer with color coded hand skeletons
- Frame by frame navigation using keyboard arrow keys
- Automatic discovery of recording labels

## Hand Landmark Representation

MediaPipe detects 21 landmarks per hand, each containing x, y, and z coordinates.

![MediaPipe hand landmarks](./assets/hand-landmarks.png)

The project combines wrist relative landmark positions to represent handshape with absolute wrist coordinates to preserve spatial movement information.

## Landmark Replay

Built an interactive replay viewer that reconstructs hand skeletons from saved numerical features. This allows visual inspection of recorded movements without storing the original webcam video.

| Frame 1/30 | Frame 30/30 |
|---|---|
| ![Frame 1](./assets/replay-frame-01.png) | ![Frame 30](./assets/replay-frame-30.png) |

## Initial Labels

Currently working with three labels: **A**, **J**, and **NAME**.

| Letter A | Letter J |
|---|---|
| ![ASL A](./assets/asl-a.png) | ![ASL J](./assets/asl-j.png) |

## Technical Challenges

- **Handshape vs. movement:** Wrist relative coordinates help represent handshape independently of camera position, while absolute wrist positions preserve movement information.
- **Data quality:** Developed a frame by frame replay viewer to inspect recorded landmark sequences before model training.

## Next Steps

- Expand the training dataset across all labels
- Train a PyTorch LSTM on 30 frame landmark sequences
- Evaluate model performance using accuracy and a confusion matrix
- Integrate real time webcam predictions

> **Status:** Data collection and landmark replay implemented. LSTM training and live sign classification are upcoming.