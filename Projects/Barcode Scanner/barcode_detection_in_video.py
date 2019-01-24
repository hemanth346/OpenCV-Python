
#Work in progress

import numpy as np
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help="Path to video file, Connected camera device will be taken as default")
args = vars(ap.parse_args())

#choosing capture method
vid = cv2.VideoCapture(args["video"]) if args.get("video",True) else cv2.VideoCapture(0)

if not (vid.isOpened()):
    print("Error reading video feed")

while (vid.isOpened()):
    check, frame = vid.read()
    # to stop after video file is completed
    if check is False:
        break
    img = frame.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #using Scharr operator to find gradients
    xGrad = cv2.Sobel(gray,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    yGrad = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=-1)
    gradient = cv2.subtract(xGrad,yGrad) # to get regions with high horizontal and low vertical gradients
    gradient = cv2.convertScaleAbs(gradient)
    #blur and threshold
    blurred = cv2.blur(gradient,(9,9))
    (_,threshold) = cv2.threshold(blurred,225,255,cv2.THRESH_BINARY)
    #gblurred = cv2.GaussianBlur(gradient,(9,9),0)
    #(_,gthreshold) = cv2.threshold(gblurred,225,255,cv2.THRESH_BINARY)


    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()

#    cv2.imshow("Original",img)
#    cv2.imshow("gradient",gradient)
#    cv2.imshow("Blurred",blurred)
#    cv2.imshow("Threshold",threshold)
#    cv2.imshow("GBlurred",gblurred)
#    cv2.imshow("GThreshold",gthreshold)
