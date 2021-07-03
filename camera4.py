import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

detector = htm.handDetector(detectionCon=0.75, maxHands=1)

class VideoCamera4(object):
    def __init__(self):
        # capturing video
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.pTime = 0
        self.smoothness = 10

        # Volume Initialization
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

        # Volume Variables
        volRange = self.volume.GetVolumeRange()
        self.minVol = volRange[0]
        self.maxVol = volRange[1]
        self.minDist = 20  # 50    # 50
        self.maxDist = 200  # 300   # 200
        self.vol = 0
        self.volBar = 400
        self.volPer = 0
        self.area = 0
        self.colorVol = (255, 0, 0)

    def __del__(self):
        pass
        # releasing camera
        self.cap.release()

    def get_frame(self):
        # extracting frames

        success, img = self.cap.read()
        img = cv2.flip(img, 1)

        # Find Hand
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=True)
        if len(lmList) != 0:

            # Filter based on size
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
            # print(area)
            if 250 < area < 1000:

                # Find Distance between index and Thumb
                length, img, lineInfo = detector.findDistance(4, 8, img)
                # print(length)

                # Convert Volume
                self.volBar = np.interp(length, [self.minDist, self.maxDist], [400, 150])
                self.volPer = np.interp(length, [self.minDist, self.maxDist], [0, 100])

                # Reduce Resolution to make it smoother
                self.volPer = self.smoothness * round(self.volPer / self.smoothness)

                # Check fingers up
                fingers = detector.fingersUp()
                # print(fingers)

                # If pinky is down set volume
                if not fingers[4]:
                    self.volume.SetMasterVolumeLevelScalar(self.volPer / 100, None)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    self.colorVol = (0, 255, 0)
                else:
                    self.colorVol = (255, 0, 0)

        # Drawings
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(self.volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(self.volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cVol = int(self.volume.GetMasterVolumeLevelScalar() * 100)
        cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, self.colorVol, 3)

        # Frame rate
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        # cv2.imshow("Img", img)
        # cv2.waitKey(1)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()