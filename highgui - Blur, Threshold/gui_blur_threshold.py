import numpy as np
import cv2
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-i","--image",required=True,help="Provide path to image")
args = vars(parse.parse_args())


try:
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    global_blurred = cv2.blur(gray,(2,2))
    _,global_thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)

    screen_res = 1280, 720
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    out_width = int(image.shape[1] * scale)
    out_height = int(image.shape[0] * scale)
    output_dim = (out_width, out_height)
except:
    exit("Please check filename or if only image is given as input")

def blur_img(image):
    '''
    Applies different kind of image smoothing techniques
    Inputs kernel size and blur type obtained from Trackbar
    '''
    ksize = cv2.getTrackbarPos(title_trackbar_kernel,title_window_blur)
    ksize = (2*ksize)+1
    type = cv2.getTrackbarPos(title_trackbar_blur,title_window_blur)
    if type == 0:
        blurred = cv2.boxFilter(gray, -1, (ksize,ksize) ,normalize=False)
        name = 'Averaging or boxFilter'
    elif type == 1:
        blurred = cv2.blur(gray,(ksize,ksize))
        name = 'Normalized boxFilter'
    elif type == 2:
        blurred = cv2.GaussianBlur(gray,(ksize,ksize),0)
        name = 'Gaussian Blur (sigma=0)'
    elif type == 3:
        blurred = cv2.medianBlur(gray,ksize)
        name = 'Median Blur'
    elif type == 4:
        pixel_diameter = 9
        sigma_color = 100
        sigma_space = 100
        blurred = cv2.bilateralFilter(gray, pixel_diameter, sigma_space, sigma_color)
        name = 'Bilateral Filter'
    global global_blurred
    global_blurred = blurred.copy()
    text = name + '-kernel:' + str(ksize)
    #cv2.putText(img, text , (20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(192,234,123),1)
    return blurred,text

def thresh_img(image):
    '''
    Applies different kind of thresholdings
    Inputs threshold value and threshold type obtained from Trackbar
    '''
    thresh_value = cv2.getTrackbarPos(title_trackbar_thresh_value, title_window_thresh)
    type = cv2.getTrackbarPos(title_trackbar_thresh_type, title_window_thresh)
    if type == 0:
        ret, thresh = cv2.threshold(image,thresh_value,255,cv2.THRESH_BINARY)
        name = 'THRESH_BINARY'
    elif type == 1:
        ret, thresh = cv2.threshold(image,thresh_value,255,cv2.THRESH_BINARY_INV)
        name = 'THRESH_BINARY_INV'
    elif type == 2:
        ret, thresh = cv2.threshold(image,thresh_value,255,cv2.THRESH_TRUNC)
        name = 'THRESH_TRUNC'
    elif type == 3:
        ret, thresh = cv2.threshold(image,thresh_value,255,cv2.THRESH_TOZERO)
        name = 'THRESH_TOZERO'
    elif type == 4:
        ret, thresh = cv2.threshold(image,thresh_value,255,cv2.THRESH_TOZERO_INV)
        name = 'THRESH_TOZERO_INV'
    elif type == 5:
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        block_size = 11
        constant = 2
        thresh = cv2.adaptiveThreshold(image,255,adaptive_method,cv2.THRESH_BINARY, block_size, constant)
        name = 'Adaptive thresholding - Mean'
    elif type == 6:
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        block_size = 11
        constant = 2
        thresh = cv2.adaptiveThreshold(image,255,adaptive_method,cv2.THRESH_BINARY,block_size, constant)
        name = 'Adaptive thresholding - Gaussian'
    elif type == 7:
        ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        name = "Otsu's Binarization"
    global global_thresh
    global_thresh = thresh.copy()
    return thresh, name

def blur_track(val):
    '''
    Function call everytime value is changed in Trackbar
        In order to facilate combined effect of image blur and thresholding
    '''
    output,text = blur_img(gray)
    output = cv2.resize(output, output_dim)
    cv2.putText(output, text, (int(out_width * 0.1), int(out_height*0.2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
    cv2.imshow(title_window_blur,output)
    combined()

def thresh_track(val):
    '''
    Function call everytime value is changed in Trackbar
        In order to facilate combined effect of image blur and thresholding
    '''
    output,name = thresh_img(gray)
    output = cv2.resize(output, output_dim)
    cv2.putText(output, name, (int(out_width * 0.1), int(out_height*0.2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
    cv2.imshow(title_window_thresh,output)
    combined()

def combined():
    '''
    Display combined smoothing and thresholding effect on images
    '''
    thresh_img(global_blurred)
    output = cv2.resize(global_thresh, output_dim)
    cv2.imshow(title_window_combined, output)

title_window_blur = "Play with blur"
title_window_thresh = "Play with threshold"
title_window_combined = "Combined effect"

title_trackbar_blur = "Blur : "
title_trackbar_kernel = "Kernel (2n+1): "
title_trackbar_thresh_type = "Type : "
title_trackbar_thresh_value = "Value : "

max_blur_types = 4
max_kernel_size = 10
max_thresh_types = 7
max_thresh_value = 255

cv2.namedWindow(title_window_blur,cv2.WINDOW_NORMAL)
cv2.createTrackbar(title_trackbar_blur, title_window_blur, 2, max_blur_types, blur_track)
cv2.createTrackbar(title_trackbar_kernel, title_window_blur, 1, max_kernel_size, blur_track)

cv2.namedWindow(title_window_thresh,cv2.WINDOW_NORMAL)
cv2.createTrackbar(title_trackbar_thresh_type, title_window_thresh, 1, max_thresh_types, thresh_track)
cv2.createTrackbar(title_trackbar_thresh_value, title_window_thresh, 128, max_thresh_value, thresh_track)

cv2.namedWindow(title_window_combined,cv2.WINDOW_NORMAL)

cv2.waitKey(0)
cv2.destroyAllWindows()
