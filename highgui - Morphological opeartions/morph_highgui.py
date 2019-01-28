import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description = "Code for various morphological operations")
parser.add_argument("-i","--image",required=True,help="Path to input image")
args = vars(parser.parse_args())

try:
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray scaled image", gray)
    blurred = cv2.GaussianBlur(gray,(9,9),0)

    width = image.shape[1]
    height = image.shape[0]
    output_dim = (width, height)
except:
    exit("Please check filename or if only image is given as input")

def morphex(val):
    (_, thresh) = cv2.threshold(blurred,128,255,cv2.THRESH_BINARY)
    get_morphed_image(thresh,morph_window)

def morphex2(val):
    (_, thresh) = cv2.threshold(blurred,128,255,cv2.THRESH_BINARY)
    get_morphed_image(thresh,morph_window2)

def morphex3(val):
    (_, thresh) = cv2.threshold(blurred,128,255,cv2.THRESH_BINARY)
    get_morphed_image(image,morph_window3)


def get_morphed_image(thresh,morph_window):
    '''
    Gets morphological operators, kernel shape and kernel size from trackbar
    and apply respective transformations to image.

    Takes two arguments - image, namedWindow
    '''

    ksize = cv2.getTrackbarPos(title_trackbar_kersize, morph_window)

    kernel_type = cv2.getTrackbarPos(title_trackbar_kershape, morph_window)
    kernel_type_dict = {0:cv2.MORPH_RECT, 1:cv2.MORPH_CROSS, 2:cv2.MORPH_ELLIPSE}
    shape = kernel_type_dict[kernel_type]
    kernel = cv2.getStructuringElement(shape, (2*ksize + 1, 2*ksize+1), (ksize, ksize))

    #shape1 = kernel_type_dict[1]
    #kernel1 = cv2.getStructuringElement(shape1, (2*ksize + 1, 2*ksize+1), (ksize, ksize))

    morph_type = cv2.getTrackbarPos(title_trackbar_morphtype, morph_window)
    if morph_type == 0:
        img = cv2.erode(thresh, kernel)
        type = 'Erode'
    elif morph_type == 1:
        img = cv2.dilate(thresh, kernel)
        type = 'Dilate'
    elif morph_type == 2:
        img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        type = 'Open'
    elif morph_type == 3:
        img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        type = 'Close'
    elif morph_type == 4:
        img = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        type = 'Gradient'
    elif morph_type == 5:
        img = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)
        type = 'Top hat'
    elif morph_type == 6:
        img = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)
        type = 'Blackhat'
    kernel_name = ['-Rect-','-Cross-','-Ellipse-']
    text = type + kernel_name[shape] + str(ksize)
    cv2.putText(img, text , (20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(192,234,123),1)
    output = cv2.resize(img, output_dim)
    cv2.imshow(morph_window,output)


max_types = 6
max_shapes = 2
max_size = 21
title_trackbar_morphtype = 'Morphological transformations : '
title_trackbar_kershape = "Structuring elements : "
title_trackbar_kersize = "Size of kernel: \n 2n+1"
#title_trackbar_morphtype = 'Morph types :\n 0: Erode \n 1: Dilate \n 2: Open \n 3: Close \n 4: Gradient \n 5: Top hat \n 6: Black hat'
#title_trackbar_kershape = "Element : \n 0: Rect \n 1: Cross \n 2: Ellipse"
#title_trackbar_kersize = "Kernel size: \n 2n+1"

morph_window = "Play with morphological Transforms"
cv2.namedWindow(morph_window, cv2.WINDOW_NORMAL)

cv2.createTrackbar(title_trackbar_morphtype, morph_window, 0, max_types, morphex)
cv2.createTrackbar(title_trackbar_kershape, morph_window, 0, max_shapes, morphex)
cv2.createTrackbar(title_trackbar_kersize, morph_window, 0, max_size, morphex)

morph_window2 = "Play with morphological Transforms_2"
cv2.namedWindow(morph_window2, cv2.WINDOW_NORMAL)

cv2.createTrackbar(title_trackbar_morphtype, morph_window2, 0, max_types, morphex2)
cv2.createTrackbar(title_trackbar_kershape, morph_window2, 0, max_shapes, morphex2)
cv2.createTrackbar(title_trackbar_kersize, morph_window2, 0, max_size, morphex2)

morph_window3 = "Play with morphological Transforms_3"
cv2.namedWindow(morph_window3, cv2.WINDOW_NORMAL)

cv2.createTrackbar(title_trackbar_morphtype, morph_window3, 0, max_types, morphex3)
cv2.createTrackbar(title_trackbar_kershape, morph_window3, 0, max_shapes, morphex3)
cv2.createTrackbar(title_trackbar_kersize, morph_window3, 0, max_size, morphex3)

cv2.waitKey(0)
cv2.destroyAllWindows()
