import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

'''while True:
    ret,img = cap.read()
    cv2.imshow('img', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
'''
while True:
    ret, img = cap.read()
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 8)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        eroi_gray = gray[int(y+h/4):int(y+3*h/4), x:x+w]
        eroi_color = img[int(y+h/4):int(y+3*h/4), x:x+w]
        eyes = eye_cascade.detectMultiScale(eroi_gray, 1.3, 15)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(eroi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            proi_gray = eroi_gray[ey:ey+eh, ex:ex+ew]
            proi_color = eroi_color[ey:ey+eh, ex:ex+ew]
            ret, proi_mask = cv2.threshold(proi_gray, 50, 255,
                    cv2.THRESH_BINARY)
            proi_gray = cv2.bitwise_and(proi_gray, proi_gray, mask=proi_mask)
            proi_gray[proi_mask > 0] = 255
            circles = cv2.HoughCircles(proi_gray, cv2.HOUGH_GRADIENT, 1, 20,
                    param1 = 15, param2 = 10, minRadius = 0, maxRadius = int(ew/4))
            #print(eh)
            if circles is None:
                break
            circles = np.uint16(np.around(circles))
            #cv2.imshow('foo', proi_gray)
            for i in circles[0,:]:
                cv2.circle(proi_color, (i[0], i[1]), i[2], (255,0,0), 2)
                cv2.circle(proi_color,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
'''
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #can alter these constants
    for (x,y,w,h) in faces:
        roi_gray1 = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray1)
        for (ex,ey,ew,eh) in eyes:
            roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
            ret, mask = cv2.threashold(img2gray, 220, 255, cv2.THRESH_BINARY_INV) #can alter these constants
            detectIris(mask)
            #draw points at iris
    cv2.imshow('img', img)
    k = cv2.watiKey(30) & 0xff
    if k == 27:
        break
'''

cap.release()
cv2.destroyAllWindows()
