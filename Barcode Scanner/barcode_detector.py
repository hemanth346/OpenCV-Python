import numpy as np
import cv2
import sys


help = '''
Usage:
python3 barcode_detector.py filename
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


def get_contours(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
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
    # cv2.imshow("closed",closed)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, None, iterations = 9)    #to remove small unnecessary blobs; erosion followed by dilation
    # cv2.imshow("opened",opened)

    contours = cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 2 :
        # OpenCV v2.4, v4-beta, or v4-official
        contours = contours[0]
    elif len(contours) == 3:
        # OpenCV v3, v4-pre, or v4-alpha
        contours = contours[1]
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature again. Refer to OpenCV's documentation "
            "in that case"))
    return contours

def draw_contours(image, cnts):
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 128, 193), 3)
    return image

if file_type == "image":
    image = cv2.imread(data_file)
    cnts = get_contours(image)
    #fetching largest contour, making it list to pass to draw_contours function
    large_cnt = [sorted(cnts,key=cv2.contourArea,reverse=True)[0]]
    display = draw_contours(image,large_cnt)
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
        cnts = get_contours(image)
        display = draw_contours(image,cnts)
        cv2.imshow("image", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
