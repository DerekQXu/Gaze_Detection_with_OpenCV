import cv2
import numpy as np
import statistics
from collections import deque

class Coord:
    def __init__(self):
        self.x = 0
        self.y = 0
    def getX(self):
        return self.x
    def getY(self):
        return self.Y

class PupilDetection:
    def __init__(self):
        self.left_pupil_q = deque()
        self.right_pupil_q = deque()
        self.q_size = 0
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.cap = cv2.VideoCapture(0)
        self.face = []
        self.eyes = []
        self.gray
        self.eroi_gray
        self.proi_gray
    def __del__(self):
        cap.release()
        cv2.destroyAllWindows()
    def getFace(self):
        ret, img = self.cap.read()
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        img = cv2.medianBlur(img, 5)
        self.gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        self.faces = face_cascade.detectMultiScale(gray, 1.3, 8)
    def getEyes(self):
        self.getFace()
        for (x,y,w,h) in faces:
            self.eroi_gray = cv2.medianBlur(self.gray[int(y+h/5):int(y+h/1.8), x:x+w], 5)
            #detect eyes
            self.eyes = eye_cascade.detectMultiScale(self.eroi_gray, 1.3, 5)
    def getPupils(self):
        self.getEyes()
        if len(self.eyes) != 2:
            print("cannot detect 2 eyes")
            return
        eye1 = Coord()
        eye2 = Coord()
        self.q_size += 1
        count = 0
        for (x,y,w,h) in eyes:
            #setup
            proi_gray = self.eroi_gray[y:y+h, x:x+w]
            #threshold
            minbrightness = 255
            for i in range(0, ew-1):
                for j in range(0, int((eh-1)*3/4)):
                    if proi_gray[i,j] < minbrightness:
                        minbrightness = proi_gray[i,j]
            ret, proi_mask = cv2.threshold(proi_gray, (minbrightness+2), 255,
                    cv2.THRESH_BINARY)
            proi_gray = cv2.bitwise_and(proi_gray, proi_gray, mask=proi_mask)
            #average blob coordinates
            listX = []
            listY = []
            for px in range(0, ew):
                for py in range(0, eh):
                    if proi_gray[px,py] < 50:
                        listX.append(px)
                        listY.append(py)
            #TODO: change this
            if count == 0:
                eye1.x = int(statistics.median(listX))
                eye1.y = int(statistics.median(listY))
            else:
                eye2.x = int(statistics.median(listX))
                eye2.y = int(statistics.median(listY))
        #TODO: change implementation of this
        if eye1.getX() < eye2.getY():
            self.left_pupil_q.append(eye1)
            self.right_pupil_q.append(eye2)
        else:
            self.left_pupil_q.append(eye2)
            self.right_pupil_q.append(eye1)
        if self.q_size > 6:
            self.left_pupil_q.popleft()
            self.right_pupil_q.popLeft()
            self.q_size -= 1
