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
    #gblurred = cv2.GaussianBlur(gradient,(9,9),0)
    #(_,gthreshold) = cv2.threshold(gblurred,128,255,cv2.THRESH_BINARY)
    blurred = cv2.blur(gradient,(9,9))
    (_,threshold) = cv2.threshold(blurred,225,255,cv2.THRESH_BINARY)
    #to close the gaps and get a rectanglur block over code
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 9))    #using kernel with high width to be able to get some distance bars in barcode
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, None, iterations = 9)    #to remove small unnecessary blobs; erosion followed by dilation
    _,cnts,_ = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], -1, (0, 128, 193), 3)
    cv2.imshow("Image", img)
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
#    cv2.imshow("Closed",closed)
#    cv2.imshow("Opened",opened)
