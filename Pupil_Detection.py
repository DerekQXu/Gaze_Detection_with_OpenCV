import cv2
import numpy as np
import statistics
from collections import deque

#coord class used to store pupil Coordinates
class Coord:
    def __init__(self):
        self.x = 0
        self.y = 0
    def getX(self):
        return self.x
    def getY(self):
        return self.y

#pupil detection class
class PupilDetection:
    def __init__(self):
        #make queues of coords for each pupil
        self.left_pupil_q = deque()
        self.right_pupil_q = deque()
        #used as offset by face and eye cascade
        self.pupil_offset1 = Coord()
        self.pupil_offset2 = Coord()
        #initialize the queues
        self.left_pupil_q.append(self.pupil_offset1)
        self.right_pupil_q.append(self.pupil_offset2)
        self.q_size = 0
        #initialize the Haar Cascades
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        #begin video capture
        self.cap = cv2.VideoCapture(0)
        #get webcam statistics
        ret, temp = self.cap.read()
        self.height, self.width, self.channels = temp.shape
        self.face = []
        self.eyes = []
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
    def getFace(self):
        #retrieve webcam data
        ret, self.color = self.cap.read()
        #apply blurring
        img = cv2.medianBlur(self.color, 5)
        #apply CLAHE (handles lighting issues)
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        self.gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        #face cascade
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 8)
    def getEyes(self):
        #get face locale first
        self.getFace()
        for (x,y,w,h) in self.faces:
            #set offsets
            self.pupil_offset1.x = x
            self.pupil_offset1.y = int(y+h/5)
            self.pupil_offset2.x = x
            self.pupil_offset2.y = int(y+h/5)
            #apply another blur
            self.eroi_gray = cv2.medianBlur(self.gray[int(y+h/5):int(y+h/1.8), x:x+w], 5)
            #eye cascade
            self.eyes = self.eye_cascade.detectMultiScale(self.eroi_gray, 1.3, 5)
    #NOTE: features to add: choose custom threshold, Haar Cascade parameters.
    def getPupils(self):
        #get eye locale first
        self.getEyes()
        #initialize temp eye variables
        eye1 = Coord()
        eye2 = Coord()
        #increment queue size
        self.q_size += 1
        #TODO: Check for 2 eyes -> less false readings from faulty eye detection
        #count used to track for loop iteration
        count = 0
        for (x,y,w,h) in self.eyes:
            #set offsets
            if count == 0:
                self.pupil_offset1.x += x
                self.pupil_offset1.y += y
            elif count == 1:
                self.pupil_offset2.x += x
                self.pupil_offset2.y += y
            else:
                break
            proi_gray = self.eroi_gray[y:y+h, x:x+w]
            #get minimum brightness (usually the pupil)
            minbrightness = 255
            for i in range(0, w):
                for j in range(0, int(h*3/4)):
                    if proi_gray[i,j] < minbrightness:
                        minbrightness = proi_gray[i,j]
            #standard mask
            mask = cv2.inRange(proi_gray, int(minbrightness), int(minbrightness)+10)
            #processed mask
            mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
            #alternate mask
            ret, proi_mask = cv2.threshold(proi_gray, (minbrightness+10), 255,
                    cv2.THRESH_BINARY)
                #NOTE: you may choose whichever mask to preference:
                # standard mask = faster; less accuracy
                # processed mask = slower; more accuracy (recommended)
            ''' For debugging:
            cv2.imshow("std", mask_open)
            cv2.imshow("proc", mask)
            cv2.imshow("alt", proi_mask)
            '''
            #get center of mask
            contours = cv2.findContours(mask_open, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
            if len(contours) == 0:
                continue
            moments = cv2.moments(contours[0])
            if moments['m00'] == 0:
                continue
            #compute x and y of pupil
            if count == 0:
                eye1.x = self.pupil_offset1.x + int(moments['m01']/moments['m00'])
                eye1.y = self.pupil_offset1.y + int(moments['m10']/moments['m00'])
            else:
                eye2.x = self.pupil_offset2.x + int(moments['m01']/moments['m00'])
                eye2.y = self.pupil_offset2.y + int(moments['m10']/moments['m00'])
            #iterate
            count += 1
        #print relevant eye location information
        print("===================")
        print("eye1 locale: (%d, %d)" % (eye1.x, eye1.y))
        print("eye2 loclae: (%d, %d)" % (eye2.x, eye2.y))
        #append eye info to the queue
        if eye1.getX() < eye2.getY():
            self.left_pupil_q.append(eye1)
            self.right_pupil_q.append(eye2)
        else:
            self.left_pupil_q.append(eye2)
            self.right_pupil_q.append(eye1)
        #keep queue at fixed size
        if self.q_size > 6:
            self.left_pupil_q.popleft()
            self.right_pupil_q.popleft()
            self.q_size -= 1
    def showPupils(self):
        #update pupil queues
        self.getPupils()
        #print out the queues
        for i in range(0, len(self.right_pupil_q)):
            self.drawPupils((self.right_pupil_q[i]).x, (self.right_pupil_q[i]).y)
        for i in range(0, len(self.left_pupil_q)):
            self.drawPupils((self.left_pupil_q[i]).x, (self.left_pupil_q[i]).y)
        #display webcam input with drawn pupil detected places
        cv2.imshow('img', self.color)
    def drawPupils(self, x, y):
        if x > self.height or x < 0:
            print ("attempting to print out of range")
            return
        if y > self.width or y < 0:
            print ("attempting to print out of range")
            return
        #draw square on given location
        for px in range(x-3, x+3):
            for py in range(y-3, y+3):
                self.color[py, px] = [0,0,255]

def main():
    #begin script
    capture = PupilDetection()
    while True:
        #show updated pupil_location
        capture.showPupils()
        #STOP executing program with Esc key
        intrrpt = cv2.waitKey(30) & 0xff
        if intrrpt == 27:
            break

if __name__ == "__main__":
    main()
