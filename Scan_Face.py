import cv2
import os
import time
import numpy as np
from PIL import Image
from time import sleep
import pickle 

face__ =cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
pretime = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"data")
check = os.path.isdir('D:/Python 2020/FaceDetection/data/viet')


def check_name(name) :
    name = 'D:/Python 2020/FaceDetection/data/' + name
    return os.path.isdir(name)

def save_data(name,img) :
    name = 'D:/Python 2020/FaceDetection/data/' + name + '/' + str(time.time()) + '.jpg'
    cv2.imwrite(name,img)

while True :
    name = input("Name = ")
    if (name == 'stop') : 
        break

    blank = np.ones(shape=[500, 500], dtype=np.uint8)
    count = 0;
    while (True) :
        ret, frame = video.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face__.detectMultiScale(
                gray,
                scaleFactor = 1.3,
                minNeighbors = 2,
                minSize =(100,100)
            )
        for (x, y, w, h) in faces :
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]
            save_data(name, roi)
            resized = cv2.resize(roi, (50,50), interpolation = cv2.INTER_AREA)
            x = int(count/10)*50
            y = int(count/10)*50 + 50
            w = (count%10)*50
            h = (count%10)*50+50
            print(x,y,w,h)
            blank[x:y,w:h] = resized
            cv2.imshow("roi",blank)
            count += 1

        cv2.imwrite("img.jpg",blank)
        fps = str(round(1/(time.time()-pretime),1))+" FPS"
        cv2.putText(frame, fps, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        pretime = time.time()
        cv2.imshow('Window', frame)
        sleep(0.1)
        key = cv2.waitKey(25)
        if key == ord('q') or count == 100 :
            cv2.destroyAllWindows()
            break
        if key == ord('e') :
            video.release
            exit()


video.release()
