import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Provide path to input image file")
args = vars(ap.parse_args())

file = args["image"]
image = cv2.imread(file)
##cv2.imshow("original",image)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##cv2.imshow('gray',gray)
##cv2.waitKey(1000)
edged = cv2.Canny(gray, 20, 100) 		#Edge detection
##cv2.imshow("edged", edged)
##cv2.waitKey(1000)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]		# Thresholding
##cv2.imshow("Threshold", thresh)
##cv2.waitKey(1000)

# Detecting and drawing contours
cnt_mat = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnt_mat[1] #as we have cv version 3.4
output = image.copy()
for c in cnts:
    cv2.drawContours(output, [c], -1, (255,0,0), 3)
    cv2.imshow('Output',output)
    cv2.waitKey(500)
text = '{} blocks detected'.format(len(cnts))
cv2.putText(output,text,(30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 3)
cv2.imshow('output',output)
cv2.waitKey(3000)
cv2.destroyAllWindows()
