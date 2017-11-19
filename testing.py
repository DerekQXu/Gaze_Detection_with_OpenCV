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
        return self.y

class PupilDetection:
    def __init__(self):
        self.left_pupil_q = deque()
        self.right_pupil_q = deque()
        temp = Coord()
        self.left_pupil_q.append(temp)
        self.right_pupil_q.append(temp)
        #note in order: left_x, right_x, left_y, right_y
        self.q_size = 0
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.cap = cv2.VideoCapture(0)
        self.face = []
        self.eyes = []
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
    def getFace(self):
        ret, self.color = self.cap.read()
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        img = cv2.medianBlur(self.color, 5)
        self.gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 8)
    def getEyes(self):
        self.getFace()
        for (x,y,w,h) in self.faces:
            self.eroi_gray = cv2.medianBlur(self.gray[int(y+h/5):int(y+h/1.8), x:x+w], 5)
            self.eroi_color = self.color[int(y+h/5):int(y+h/1.8), x:x+w]
            self.eyes = self.eye_cascade.detectMultiScale(self.eroi_gray, 1.3, 5)
    def getPupils(self):
        self.getEyes()
        if len(self.eyes) != 2:
            print("cannot detect 2 eyes")
            return
        eye1 = Coord()
        eye2 = Coord()
        self.q_size += 1
        count = 0
        for (x,y,w,h) in self.eyes:
            #setup
            proi_gray = self.eroi_gray[y:y+h, x:x+w]
            #TODO: edit this
            self.proi_color = self.eroi_color[y:y+h, x:x+w]
            #threshold
            minbrightness = 255
            for i in range(0, w):
                for j in range(0, int(h*3/4)):
                    if proi_gray[i,j] < minbrightness:
                        minbrightness = proi_gray[i,j]
            ret, proi_mask = cv2.threshold(proi_gray, (minbrightness+2), 255,
                    cv2.THRESH_BINARY)
            proi_gray = cv2.bitwise_and(proi_gray, proi_gray, mask=proi_mask)
            #average blob coordinates
            listX = []
            listY = []
            for px in range(0, w):
                for py in range(0, h):
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
            self.right_pupil_q.popleft()
            self.q_size -= 1
    def showPupils(self):
        self.getPupils()
        #TODO: implement through Q:
        '''
        #for coord in self.left_pupil_q:
        coord = self.left_pupil_q[0]
        for px in range(coord.x-3, coord.x+3):
            for py in range(coord.y-3, coord.y+3):
                self.proi_color[px, py] = [0,0,255]
        '''
        #for coord in self.right_pupil_q:
        coord = self.right_pupil_q[0]
        for px in range(coord.x-3, coord.x+3):
            for py in range(coord.y-3, coord.y+3):
                self.proi_color[px, py] = [0,0,255]
        cv2.imshow('img', self.color)


def main():
    capture = PupilDetection()
    while True:
        capture.showPupils()
        intrrpt = cv2.waitKey(30) & 0xff
        if intrrpt == 27:
            break

if __name__ == "__main__":
    main()
