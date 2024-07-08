import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color = np.array([6, 160, 210])  # 此时的取值的橙色
    upper_color = np.array([9, 190, 240])

    color_obj = cv2.inRange(frame_hsv, lower_color, upper_color)

    conts, hrc = cv2.findContours(color_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(conts)

    frame = cv2.drawContours(frame, conts, -1, (0, 255, 0), 3)

    bigconts = []
    for cont in conts:
        area = cv2.contourArea(cont)
        if area > 200:
            bigconts.append(cont)

    for bigcnt in bigconts:
        M = cv2.moments(bigcnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(frame, (cx, cy), 100, (0, 0, 255), 5)

        frame = cv2.drawContours(frame, bigconts, -1, (255, 0, 0), 10)

    # Display the resulting frame
    frame = cv2.flip(frame, 1)
    cv2.imshow('my show windows', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
