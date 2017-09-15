import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    #setup
    ret, img = cap.read()
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    for (x,y,w,h) in faces:
        #setup
        '''uncomment for face rectangle:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)'''
        eroi_gray = gray[int(y+h/4):int(y+3*h/4), x:x+w]
        eroi_gray = cv2.medianBlur(eroi_gray, 5)
        eroi_color = img[int(y+h/4):int(y+3*h/4), x:x+w]
        #detect eyes
        eyes = eye_cascade.detectMultiScale(eroi_gray, 1.3, 15)
        for (ex,ey,ew,eh) in eyes:
            #setup
            '''uncomment for eye rectangle:
            cv2.rectangle(eroi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)'''
            proi_gray = eroi_gray[ey:ey+eh, ex:ex+ew]
            proi_color = eroi_color[ey:ey+eh, ex:ex+ew]
            #threshold
            ret, proi_mask = cv2.threshold(proi_gray, 50, 255,
                    cv2.THRESH_BINARY)
            proi_gray = cv2.bitwise_and(proi_gray, proi_gray, mask=proi_mask)
            proi_gray[proi_mask > 0] = 255
            #average blob coordinates
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
            #draw coordinate locations
            for px in range(averageX-3, averageX+3):
                for py in range(averageY-3, averageY+3):
                    proi_color[px, py] = [0, 0, 255]
    #draw face
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
