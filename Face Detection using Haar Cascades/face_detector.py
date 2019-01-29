import numpy as np
import cv2
import sys


help = '''
Usage:
python3 face_detector.py filename
The file can be an image or a video. To read video from webcam, pass empty filename.
'''

if len(sys.argv) > 2:
    print(help)
    exit()

if len(sys.argv) == 2:
    data_file = sys.argv[1]
    file_type = None
    if data_file.split(".")[-1].lower() in ["png", "jpg", "jpeg", "tiff"]:
        file_type = "image"
    elif data_file.split(".")[-1].lower() in ["mov", "avi", "mp4", "mkv"]:
        file_type = "video"
    else:
        raise Exception(("File type not identified/supported"))
else:
    data_file = 0
    file_type = "video"


cascade = '.\haarcascades\haarcascade_frontalface_default.xml'
nested_cascade = '.\haarcascades\haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(cascade)
eye_cascade = cv2.CascadeClassifier(nested_cascade)


def get_faces(image):
    '''
    Detects face, eyes and draws rectangle around them
    '''
    scale_factor = 1.3
    min_neighbors = 5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return image

if file_type == "image":
    image = cv2.imread(data_file)
    display = get_faces(image)
    cv2.imshow("image",display )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if file_type == "video":
    vid = cv2.VideoCapture(data_file)

    if not (vid.isOpened()):
        print("Error reading video feed")

    while (vid.isOpened()):
        check, frame = vid.read()
        # to stop after video file is completed
        if check is False:
            break
        image = frame.copy()
        display = get_faces(image)
        cv2.imshow("image", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
