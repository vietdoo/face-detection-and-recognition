import time
import cv2

face__ =cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

pretime = time.time()
while True :
    ret, frame = video.read()
    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face__.detectMultiScale(
            gray,
            scaleFactor = 1.3,
            minNeighbors = 2,
            minSize =(50,50)
        )
    for (x, y, w, h) in faces :
      #  cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        frame = cv2.circle(frame, (x,y), 5, (0, 255, 0), -1)
        frame = cv2.circle(frame, (x+w,y), 5, (0, 255, 0), -1)
        frame = cv2.circle(frame, (x+20,y+h), 5, (0, 255, 0), -1)
        frame = cv2.circle(frame, (x+w-20,y+h), 5, (0, 255, 0), -1)
    fps = str(round(1/(time.time()-pretime),1))+" FPS"
    cv2.putText(frame, fps, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    pretime = time.time()
    cv2.imshow('Window', frame)

    key = cv2.waitKey(25)
    if key == ord('q') :
        break
video.release()
