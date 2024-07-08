import numpy as np
import cv2

url = "http://192.168.211.191:81/stream"
cap = cv2.VideoCapture(url)

while (True):
    ret, img = cap.read()
    H, W, C = img.shape
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        if x + w * 0.5 < W * 0.5 - 20:
            print('R')
        elif x + w * 0.5 > W * 0.5 + 20:
            print('L')
        else:
            print('C')
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    h_img = cv2.flip(img,1)
    cv2.imshow('img', h_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break