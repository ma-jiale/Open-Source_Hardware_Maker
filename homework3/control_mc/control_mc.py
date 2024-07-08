import numpy as np
import cv2
from mcpi.minecraft import Minecraft

mc = Minecraft.create()
# 调用电脑内置摄像头
cap = cv2.VideoCapture(0)
pos = mc.player.getTilePos()

while (True):
    pos = mc.player.getTilePos()
    ret, img = cap.read()
    H, W, C = img.shape
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # 通过检测人脸的位置控制人物前进后退
    if x + w * 0.5 < W * 0.5 - 30:
        if h < 250:
            print('BR')
            new_pos = pos.clone()
            new_pos.z += 1
            new_pos.x -= 1
        elif h > 190:
            print('FR')
            new_pos = pos.clone()
            new_pos.z += 1
            new_pos.x += 1
        else:
            print('R')
            new_pos = pos.clone()
            new_pos.z += 1
    elif x + w * 0.5 > W * 0.5 + 30:
        if h < 250:
            print('BL')
            new_pos = pos.clone()
            new_pos.z -= 1
            new_pos.x -= 1
        elif h > 190:
            print('FL')
            new_pos = pos.clone()
            new_pos.z -= 1
            new_pos.x += 1
        else:
            print('L')
            new_pos = pos.clone()
            new_pos.z -= 1
    else:
        if h < 170:
            print('B')
            new_pos = pos.clone()
            new_pos.x -= 1
        elif h > 270:
            print('F')
            new_pos = pos.clone()
            new_pos.x += 1
        else:
            new_pos = pos.clone()
    mc.player.setTilePos(new_pos)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    h_img = cv2.flip(img, 1)
    cv2.imshow('img', h_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
