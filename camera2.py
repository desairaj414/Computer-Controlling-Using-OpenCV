import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

detector = htm.handDetector(detectionCon=0.75, maxHands=1)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

class VideoCamera2(object):
    def __init__(self):
        # capturing video
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.pTime = 0
        self.header = overlayList[0]
        self.drawColor = (255, 0, 255)
        self.xp, self.yp = 0, 0
        self.brushThickness = 15
        self.eraserThickness = 100

    def __del__(self):
        pass
        # releasing camera
        self.cap.release()

    def get_frame(self):
        # extracting frames

        # 1. Import image
        success, img = self.cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1280, 720))
        # print("1", img.shape)
        # print("2", imgCanvas.shape)

        # 2. Find Hand Landmarks
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)

        if len(lmList) == 0:
            self.xp, self.yp = 0, 0

        if len(lmList) != 0:
            # print(lmList)

            # tip of index and middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # 3. Check which fingers are up
            fingers = detector.fingersUp()
            # print(fingers)

            # 4. If Selection Mode - Two finger are up
            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                # print("Selection Mode")
                # # Checking for the click
                if y1 < 125:
                    if 250 < x1 < 450:
                        self.header = overlayList[0]
                        self.drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        self.header = overlayList[1]
                        self.drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        self.header = overlayList[2]
                        self.drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        self.header = overlayList[3]
                        self.drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)

            # 5. If Drawing Mode - Index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, self.drawColor, cv2.FILLED)
                # print("Drawing Mode")
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1

                if self.drawColor == (0, 0, 0):
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
                    cv2.line(imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
                else:
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)
                    cv2.line(imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)

                self.xp, self.yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgThres = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgThres, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Setting the header image
        img[0:125, 0:1280] = self.header
        # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
        # cv2.imshow("Image", img)
        # cv2.imshow("Canvas", imgCanvas)
        # cv2.imshow("Gray", imgGray)
        # cv2.imshow("Thres", imgThres)
        # cv2.imshow("Inv", imgInv)
        # cv2.waitKey(1)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()