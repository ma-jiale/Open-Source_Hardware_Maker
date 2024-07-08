# 作业要求：完成摄像头利用mediapipe检测手势，
# 完成用拇指食指指尖拿捏移动屏幕上的红色圆圈到另一个位置的功能然后放手的功能

# 1.摄像头利用mediapipe检测手势
# 2.在画面上绘制一个红色的圆，圆心的位置随机
# 3.获取食指指尖和拇指指尖的坐标
# 4.当食指指尖和拇指指尖连线中点进入圆范围并且直线长度小于一定值时，将圆的圆心改为食指指尖和拇指指尖连线中点

# STEP 1: Import the necessary modules.
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = int(width / 2)
center_y = int(height / 2)
radius = 30
distance = -1
midpoint_x = 1
midpoint_y = 1
circle_center_x = int(width / 2)
circle_center_y = int(height / 2)


def draw_circle_on_image(frame, detection_result):
    height, width, _ = frame.shape
    radius = 20
    global circle_center_x
    global circle_center_y
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Calculate the center of the circle
        thumb_tip = hand_landmarks[4]
        index_finger_tip = hand_landmarks[8]
        center_x = (thumb_tip.x + index_finger_tip.x) / 2
        center_y = (thumb_tip.y + index_finger_tip.y) / 2

        distance = math.sqrt((thumb_tip.x * width - index_finger_tip.x * width) ** 2 + (
                thumb_tip.y * height - index_finger_tip.y * height) ** 2)

        # 若符合条件更新圆坐标
        if distance <= radius*2 and circle_center_x - radius < int(
                center_x * width) < circle_center_x + radius and circle_center_y - radius < int(
                center_y * height) < circle_center_y + radius:
            circle_center_x = int(center_x * width)
            circle_center_y = int(center_y * height)

    cv2.circle(frame, (circle_center_x, circle_center_y), radius, (0, 255, 0), -1)
    return frame


# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)
with HandLandmarker.create_from_options(options) as detector:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # STEP 3: Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect_for_video(mp_image, int(frame_timestamp_ms))

        # STEP 5: Process the classification result.
        frame = draw_circle_on_image(frame, detection_result)

        frame = cv2.flip(frame, 1)
        cv2.imshow('hand landmark detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
