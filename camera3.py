import cv2
import numpy as np
import time
import os
import autopy
import HandTrackingModule as htm

detector = htm.handDetector(detectionCon=0.75, maxHands=1)

class VideoCamera3(object):
    def __init__(self):
        # capturing video
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.frameR = 100  # Frame Reduction
        self.smoothening = 7
        self.pTime = 0
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.wScr, self.hScr = autopy.screen.size()

    def __del__(self):
        pass
        # releasing camera
        self.cap.release()

    def get_frame(self):
        # extracting frames

        success, img = self.cap.read()
        img = cv2.flip(img, 1)
        # img = cv2.resize(img, (1280, 720))

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)

            # 3. Check which fingers are up
            fingers = detector.fingersUp()
            # print(fingers)
            cv2.rectangle(img, (self.frameR, self.frameR), (self.wCam - self.frameR, self.hCam - self.frameR),
                          (255, 0, 255), 2)
            # 4. Only Index Finger : Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
                y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
                # 6. Smoothen Values
                clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                clocY = self.plocY + (y3 - self.plocY) / self.smoothening

                # 7. Move Mouse
                autopy.mouse.move(clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                self.plocX, self.plocY = clocX, clocY

            # 8. Both Index and middle fingers are up : Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # 9. Find distance between fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)
                # print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        # 12. Display
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()