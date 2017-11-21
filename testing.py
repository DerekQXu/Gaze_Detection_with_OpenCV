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
        self.pupil_temp1 = Coord()
        self.pupil_temp2 = Coord()
        self.left_pupil_q.append(self.pupil_temp1)
        self.right_pupil_q.append(self.pupil_temp2)
        #note in order: left_x, right_x, left_y, right_y
        self.q_size = 0
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.cap = cv2.VideoCapture(0)
        ret, temp = self.cap.read()
        self.height, self.width, self.channels = temp.shape
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
        #print("detects %d faces" % (len(self.face)))
        #if len(self.face) != 1:
        #    return
        for (x,y,w,h) in self.faces:
            self.pupil_temp1.x = x
            self.pupil_temp1.y = int(y+h/5)
            self.pupil_temp2.x = x
            self.pupil_temp2.y = int(y+h/5)
            self.eroi_gray = cv2.medianBlur(self.gray[int(y+h/5):int(y+h/1.8), x:x+w], 5)
            self.eyes = self.eye_cascade.detectMultiScale(self.eroi_gray, 1.3, 5)
    def getPupils(self):
        self.getEyes()
        #print("detects %d eyes" % (len(self.eyes)))
        #if len(self.eyes) != 2:
        #    return
        eye1 = Coord()
        eye2 = Coord()
        self.q_size += 1
        count = 0
        for (x,y,w,h) in self.eyes:
            if count == 0:
                self.pupil_temp1.x += x
                self.pupil_temp1.y += y
            else:
                self.pupil_temp2.x += x
                self.pupil_temp2.y += y
            #setup
            proi_gray = self.eroi_gray[y:y+h, x:x+w]
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
                        listX.append(py)
                        listY.append(px)
            #TODO: change this
            if count == 0:
                eye1.x = self.pupil_temp1.x + int(statistics.median(listX))
                eye1.y = self.pupil_temp1.y + int(statistics.median(listY))
            else:
                eye2.x = self.pupil_temp2.x + int(statistics.median(listX))
                eye2.y = self.pupil_temp2.y + int(statistics.median(listY))
            count += 1
        #TODO: change implementation of this
        print("===================")
        print("eye1 locale: (%d, %d)" % (eye1.x, eye1.y))
        print("eye2 loclae: (%d, %d)" % (eye2.x, eye2.y))
        print("===================")
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
        for i in range(0, len(self.right_pupil_q)):
            self.drawPupils((self.right_pupil_q[i]).x, (self.right_pupil_q[i]).y)
        for i in range(0, len(self.left_pupil_q)):
            self.drawPupils((self.left_pupil_q[i]).x, (self.left_pupil_q[i]).y)
        cv2.imshow('img', self.color)
    def drawPupils(self, x, y):
        if x > self.height or x < 0:
            print ("attempting to print out of range")
            return
        if y > self.width or y < 0:
            print ("attempting to print out of range")
            return
        for px in range(x-3, x+3):
            for py in range(y-3, y+3):
                self.color[py, px] = [0,0,255]


def main():
    capture = PupilDetection()
    while True:
        capture.showPupils()
        intrrpt = cv2.waitKey(30) & 0xff
        if intrrpt == 27:
            break

if __name__ == "__main__":
    main()
