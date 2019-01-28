import numpy as np
import cv2
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-v","--video",
            help="Provide path to image(Optional, if not provided will stream from connected webcam)")
args = vars(parse.parse_args())

cascade = '.\haarcascades\haarcascade_frontalface_default.xml'
nested_cascade = '.\haarcascades\haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(cascade)
eye_cascade = cv2.CascadeClassifier(nested_cascade)
scale_factor = 1.3
min_neighbors = 5

#choosing capture method
vid = cv2.VideoCapture(args["video"]) if args.get("video",True) else cv2.VideoCapture(0)

if not (vid.isOpened()):
    print("Error reading video feed")

while (vid.isOpened()):
    check, frame = vid.read()
    if check is False:  # to stop after video file is completed
        break
    img = frame.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
