import cv2
import random

partsHeight: int = 50
partsWidth: int = 50
size = 3
score = 0
chessboard = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def draw_grid():
    for i in range(0, size):
        for j in range(0, size):
            # 画网格
            pt0 = (int(width / 2 - partsWidth * size / 2 + j * partsWidth),
                   int(height / 2 - partsHeight * size / 2 + i * partsHeight))
            pt1 = (int(width / 2 - partsWidth * size / 2 + j * partsWidth + partsWidth),
                   int(height / 2 - partsHeight * size / 2 + i * partsHeight + partsHeight))
            cv2.rectangle(img, pt0, pt1, (255, 0, 0), 1)


def draw_source():
    for i in range(0, size):
        for j in range(0, size):
            # 如果随机数x>5才拷贝选定区域图片
            if random.randint(1, 9) > 5:
                img[int(height / 2 - partsHeight * size / 2 + i * partsHeight):int(
                    height / 2 - partsHeight * size / 2 + (i + 1) * partsHeight),
                int(width / 2 - partsWidth * size / 2 + j * partsWidth):int(
                    width / 2 - partsWidth * size / 2 + (j + 1) * partsWidth)] = source
                chessboard[i][j] = 1


def set_score(event, x, y, flags, param):
    global score
    if event == cv2.EVENT_LBUTTONDBLCLK or event == cv2.EVENT_RBUTTONDBLCLK:
        # cv2.circle(img, (x, y), 50, (255, 0, 0), 2)
        # print("%d %d" % ((x // partsWidth - 3), y // partsHeight - 2))
        row = y // partsHeight - 2
        col = x // partsWidth - 3
        if chessboard[row][col] == 1:
            score += 10
        else:
            if score >= 5:
                score -= 5



def reset():
    for i in range(0, size):
        for j in range(0, size):
            chessboard[i][j] = 0


def show_HP():
    cv2.rectangle(img, (width - 5, 5), (width - 105, 25), (0, 0, 255), 1)
    cv2.rectangle(img, (width - 105 + score, 5), (width - 105, 25), (0, 255, 0), -1)


while True:
    # Load a color image
    img = cv2.imread('dragon.jpg', cv2.IMREAD_COLOR)  # image 426(width)*319(height)
    cv2.namedWindow('dragon')
    cv2.setMouseCallback('dragon', set_score)
    height, width, color = img.shape
    source = img[50:50 + partsHeight, 120:120 + partsWidth]  # img[row, column]

    draw_source()
    draw_grid()
    show_HP()
    cv2.imshow('dragon', img)

    if cv2.waitKey(1000) == 27:
        break
    if score > 100:
        img2 = cv2.imread('fireworks.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('fireworks', img2)
        cv2.waitKey(1000)
        break
    reset()
cv2.destroyAllWindows()
