import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",help="path to image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

Xgradient = cv2.Sobel(gray,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
Ygradient = cv2.Sobel(gray,ddepth=cv2.CV_32F,dx=0,dy=1,ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(Xgradient,Ygradient)
gradient = cv2.convertScaleAbs(gradient)


blurred = cv2.blur(gradient,(9,9))
(_, thresh) = cv2.threshold(blurred,225,255,cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
closed = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

closed = cv2.erode(closed,None,iterations=4)
closed = cv2.dilate(closed,None,iterations=4)

cnts = cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] #as version is 3.4
c = sorted(cnts,key=cv2.contourArea,reverse=True)[0] #fetching largest contour

rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)

cv2.drawContours(image,[box],-1,(0,255,0),3)
cv2.imshow("image",image)
cv2.waitKey(0)

cv2.destroyAllWindows()
