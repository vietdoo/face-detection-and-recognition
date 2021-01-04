 import time
import cv2
import pickle

face__ =cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("data_train.yml") 
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
labels =  {v:k for k,v in og_labels.items()} 

video = cv2.VideoCapture(0)
print(labels)

pretime = time.time()
while True :
    ret, frame = video.read()
    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face__.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 1,
            minSize =(200,200)
        )
    for (x, y, w, h) in faces :

      #  cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        frame = cv2.circle(frame, (x,y), 7, (0, 255, 0), -1)
        #frame = cv2.circle(frame, (x+w,y), 5, (0, 255, 0), -1)
        #frame = cv2.circle(frame, (x+20,y+h), 5, (0, 255, 0), -1)
        #frame = cv2.circle(frame, (x+w-20,y+h), 5, (0, 255, 0), -1)
        roi_gray = gray[y:y+h,x:x+w]
        cv2.imshow("gray",roi_gray)
        id_, conf = recognizer.predict(roi_gray)
        if (conf < 80 and conf > 40 ):
            cv2.putText(frame, labels[id_], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            print(conf)
    
    #fps = str(round(1/(time.time()-pretime),1))+" FPS"
   # cv2.putText(frame, fps, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    pretime = time.time()
    cv2.imshow('Window', frame)

    key = cv2.waitKey(25)
    if key == ord('q') :
        break
video.release()
