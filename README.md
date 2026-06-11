# Pose Detection System

A real-time human pose detection and landmark tracking system built with MediaPipe and OpenCV. Supports both webcam input and video file processing, with angle calculation between body landmarks.

---

## Project Structure

```
├── pose_estimationV2.py     # Core Pose_Detection class (reusable module)
└── pose_estimation_Final_LIVE.py           # Main application entry point (webcam live feed)
```

---

## Features

- Real-time pose detection using MediaPipe Pose
- Landmark detection with optional filtering by landmark ID or upper-body-only mode
- Angle calculation between any three body landmarks
- FPS overlay on the video feed
- Configurable model complexity, smoothing, and confidence thresholds
- Works with live webcam or pre-recorded video files

---

## Requirements

Install dependencies with:

```bash
pip install opencv-python mediapipe numpy
```

| Package | Purpose |
|---|---|
| `opencv-python` | Video capture, display, and drawing |
| `mediapipe` | Pose estimation model |
| `numpy` | Array math and landmark storage |

---

## Usage

### Live Webcam

```bash
python pose_estimation_Final_LIVE.py 
```

Press **`q`** to quit.

### Video File (from `pose_estimationV2.py` main)

Edit the `main()` function in `pose_estimationV2.py` and change:

```python
cap = cv.VideoCapture(r'Videos\Bicep_Curl2.mov')
```

Then run:

```bash
python pose_estimationV2.py
```

---

## API Reference

### `Pose_Detection`

```python
detector = Pose_Detection(
    mode=False,                    # Static image mode (no tracking between frames)
    complexity=1,                  # Model complexity: 0 (fast) / 1 (balanced) / 2 (accurate)
    smooth_landmarks=True,         # Smooth landmark positions to reduce jitter
    enable_segmentation=False,     # Enable background segmentation mask
    smooth_segmentation=True,      # Smooth segmentation mask
    min_detection_confidence=0.5,  # Minimum confidence to detect a pose
    min_tracking_confidence=0.5,   # Minimum confidence to track a pose
    upper_body_only=False          # Restrict landmarks to upper body (IDs 0–24)
)
```

---

### `detect_person(frame, draw=True)`

Runs pose detection on a BGR frame.

```python
frame, result = detector.detect_person(frame, draw=True)
```

| Parameter | Type | Description |
|---|---|---|
| `frame` | `np.ndarray` | BGR image from OpenCV |
| `draw` | `bool` | Draw skeleton overlay on frame |

**Returns:** `(annotated_frame, mediapipe_result)`

---

### `detect_landmark(img, result, draw=True, ids=[])`

Extracts landmark positions from a pose result.

```python
landmarks = detector.detect_landmark(img, result, draw=True, ids=[11, 13, 15])
```

| Parameter | Type | Description |
|---|---|---|
| `img` | `np.ndarray` | Frame to draw on |
| `result` | `PoseLandmarks` | Result from `detect_person` |
| `draw` | `bool` | Draw red circles on detected landmarks |
| `ids` | `list[int]` | Specific landmark IDs to return (empty = all) |

**Returns:** `np.ndarray` of shape `(N, 4)` — columns are `[id, cx, cy, visibility]`

---

### `findAngle(img, pos1, pos2, pos3, draw=True)`

Calculates the angle at `pos2` formed by the vectors `pos2→pos1` and `pos2→pos3`.

```python
detector.findAngle(img, landmarks[11], landmarks[13], landmarks[15])
```

Each `pos` argument is a row from the landmarks array: `[id, cx, cy, visibility]`.

If `draw=True`, lines and circles are drawn on `img` along with the angle value in degrees.

---

## MediaPipe Landmark IDs (Key Reference)

| ID | Landmark | ID | Landmark |
|---|---|---|---|
| 0 | Nose | 16 | Right Wrist |
| 11 | Left Shoulder | 23 | Left Hip |
| 12 | Right Shoulder | 24 | Right Hip |
| 13 | Left Elbow | 25 | Left Knee |
| 14 | Right Elbow | 26 | Right Knee |
| 15 | Left Wrist | 27 | Left Ankle |

Full reference: [MediaPipe Pose Landmarks](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

---

## Example: Bicep Curl Angle Tracking

To track elbow angle for a left bicep curl, use landmarks 11 (shoulder), 13 (elbow), and 15 (wrist):

```python
frame, result = detector.detect_person(frame, draw=False)
landmarks = detector.detect_landmark(frame, result, draw=False, ids=[11, 13, 15])

if len(landmarks) == 3:
    detector.findAngle(frame, landmarks[0], landmarks[1], landmarks[2])
```
## Supported Workouts
 
### 🦾 Bicep Curl
 
Tracks elbow flexion and extension to count reps and monitor form on both arms.
 
**How it works:** The angle at the elbow joint (between the shoulder, elbow, and wrist) is measured each frame. A rep is counted when the angle drops to a fully curled position (~30–50°) and returns to a fully extended position (~160–170°).
 
**Landmarks used:**
| Side | Shoulder | Elbow | Wrist |
|---|---|---|---|
| Left | 11 | 13 | 15 |
| Right | 12 | 14 | 16 |
 
**Camera placement:** Position the camera side-on so one arm is fully visible in profile. Both arms can be tracked simultaneously if the full body is in frame.
 
---
 
### 🏋️ Incline Bench Press
 
Tracks shoulder and elbow angles to detect the top and bottom of each press rep and flag depth issues.
 
**How it works:** The angle at the shoulder (between the elbow, shoulder, and hip) is monitored to detect when the bar is lowered to the chest (larger angle) versus pressed to full extension (smaller angle). Elbow flare can also be flagged by comparing left and right elbow angles.
 
**Landmarks used:**
| Point | Left | Right |
|---|---|---|
| Shoulder | 11 | 12 |
| Elbow | 13 | 14 |
| Hip | 23 | 24 |
| Wrist | 15 | 16 |
 
**Camera placement:** Position the camera to the side of the bench at roughly shoulder height for the clearest view of the press range of motion.
 
---
 
### 🦵 Squat
 
Tracks knee and hip angles to evaluate squat depth and flag common form errors such as not reaching parallel or excessive forward lean.
 
**How it works:** The angle at the knee (between the hip, knee, and ankle) is the primary metric. A rep is counted when the knee angle drops below the parallel threshold (~90°) and returns to standing (~170°+). The hip angle (between the shoulder, hip, and knee) is used as a secondary check for torso lean.
 
**Landmarks used:**
| Point | Left | Right |
|---|---|---|
| Hip | 23 | 24 |
| Knee | 25 | 26 |
| Ankle | 27 | 28 |
| Shoulder | 11 | 12 |
 
**Camera placement:** Position the camera directly to the side at hip height to capture the full range of knee and hip motion clearly.
 
---

---

## Notes

- `detect_person` must be called before `detect_landmark` — it produces the `result` object that `detect_landmark` reads.
- When `upper_body_only=True`, only landmark IDs 0–24 are returned.
- The `findAngle` method uses the dot product formula and does not return the angle — it only draws it. Extract the value from the method if needed by modifying the return statement.
