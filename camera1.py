# camera.py
# import the necessary packages
import cv2
import time
import os
import HandTrackingModule as htm

folderPath = "FingerImages"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image,(200,200))
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))

detector = htm.handDetector(detectionCon=0.75, maxHands=1)

class VideoCamera1(object):
    def __init__(self):
        # capturing video
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.pTime = 0
        self.tipIds = [4, 8, 12, 16, 20]

    def __del__(self):
        pass
        # releasing camera
        self.cap.release()

    def get_frame(self):
        # extracting frames

        success, img = self.cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        # print(lmList)

        if len(lmList) != 0:
            fingers = []

            # Thumb
            if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)
            totalFingers = fingers.count(1)
            # print(totalFingers)
            # print fingers pos
            str1 = ' '.join(map(str, fingers))

            h, w, c = overlayList[totalFingers - 1].shape
            img[10:h + 10, 10:w + 10] = overlayList[totalFingers - 1]

            cv2.rectangle(img, (30, 225), (180, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (55, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 25)
            cv2.putText(img, str1, (23, 450), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()