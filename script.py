import cv2
import numpy as np
import statistics

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
while True:
    #setup
    ret, img = cap.read()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = cv2.medianBlur(img,5)
    gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    #detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 8)
    '''
    while len(faces) == 0:
        featureNum = featureNum - 1
        faces = face_cascade.detectMultiScale(gray, 1.3, featureNum)
        if featureNum == 0:
            break
    '''
    for (x,y,w,h) in faces:
        #setup
        '''uncomment for face rectangle:'''
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)#'''
        eroi_gray = cv2.medianBlur(gray[int(y+h/5):int(y+h/1.8), x:x+w], 5)
        eroi_color = img[int(y+h/5):int(y+h/1.8), x:x+w]
        cv2.imshow('dbug',eroi_color)
        #detect eyes
        eyes = eye_cascade.detectMultiScale(eroi_gray, 1.3, 5)
        print(len(eyes))
        for (ex,ey,ew,eh) in eyes:
            #setup
            '''uncomment for eye rectangle:'''
            cv2.rectangle(eroi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)#'''
            proi_gray = eroi_gray[ey:ey+eh, ex:ex+ew]
            proi_color = eroi_color[ey:ey+eh, ex:ex+ew]
            #threshold
            minbrightness = 255
            for i in range(0, ew-1):
                for j in range(0, int(eh*3/4)):
                    if proi_gray[i,j] < minbrightness:
                        minbrightness = proi_gray[i,j]
            ret, proi_mask = cv2.threshold(proi_gray, (minbrightness+2), 255,
                    cv2.THRESH_BINARY)
            proi_gray = cv2.bitwise_and(proi_gray, proi_gray, mask=proi_mask)
            #proi_gray[proi_mask > 0] = 255
            #find corner of eyes
            '''
            proi_gray_fix = np.float32(proi_gray)
            dst = cv2.cornerHarris(proi_gray_fix,2,3,0.04)
            dst = cv2.dilate(dst,None)
            proi_color[dst>0.01*dst.max()] = [0,255,0]
            '''
            #average blob coordinates
            listX = []
            listY = []
            for px in range(0, ew-1):
                for py in range(0, eh-1):
                    if proi_gray[px,py] < 50:
                        listX.append(px)
                        listY.append(py)
            averageX = int(statistics.median(listX))
            averageY = int(statistics.median(listY))
            '''
            count = 0
            averageX = 0
            averageY = 0
            for px in range(0, ew):
                for py in range(0, eh):
                    if proi_gray[px,py] < 50:
                        averageX += px
                        averageY += py
                        count = count+1
            if count == 0:
                break
            averageX = int(averageX/count)
            averageY = int(averageY/count)
            print (averageX)
            print (averageY)
            '''
            #draw coordinate locations
            for px in range(averageX-3, averageX+3):
                for py in range(averageY-3, averageY+3):
                    proi_color[px, py] = [0, 0, 255]
    #draw face
    cv2.imshow('img', img)
    #cv2.imshow('debug1',gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
