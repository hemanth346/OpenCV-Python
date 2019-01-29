import numpy as np
import cv2
from skimage.filters import threshold_local
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Provide path to the image that is to be scanned")
args = vars(ap.parse_args())


def ordered_rectangle_points(pts):
    '''
    Takes an array of 4 co-ordinates and returns a array with consistent ordering of the points for the rectangle

    List contains points from top left to bottom left in that order
         -> top left, top right, bottom right, bottom left
    '''
    #initializng a array of coordinates that will be ordered
    #such that the first entry is the top-left,the second entry is the top-right,
    #the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4,2), dtype="float32")
    #[0]top-left point will have the smallest sum of the co-ordinates
    #[2]bottom-right point will have largest sum
    #[1]top-right point will have smallest difference between the co-ordinates
    #[3]where as bottom-left will have largest diff

    p_sum = np.sum(pts,axis=1)
    p_diff = np.diff(pts,axis=1)
    # From top left to bottom left in that order -> top left, top right, bottom right, bottom left
    rect[0] = pts[np.argmin(p_sum)]
    rect[1] = pts[np.argmin(p_diff)]
    rect[2] = pts[np.argmax(p_sum)]
    rect[3] = pts[np.argmax(p_diff)]

    return rect


def four_point_perspective_transform(image, pts):
    '''
    Will do a 4 point perspective transform to obtain a top-down, “birds eye view” of an image

        Takes input as image and list of four reference points that contain the ROI of the image we want to transform
    '''
    # obtaining consistent order of points and unpacking them individually
    rect = ordered_rectangle_points(pts)
    (tl,tr,br,bl) = rect

    # computing the width of the new image, which will be the maximum distance between
    # bottom-right and bottom-left coordiates or
    # top-right and top-left coordinates
    widthA = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2)) # tr - tl
    widthB = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2)) # br - bl
    #maxWidth = max(widthA,widthB) # returning np.array output
    maxWidth = max(int(widthA), int(widthB))

    # computing the height of the new image, which will be the maximum distance between
    # top-right and bottom-right coordiates or
    # top-left and bottom-left coordinates
    heightA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2)) # tr - tl
    heightB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2)) # br - bl
    maxHeight = max(int(heightA), int(heightB))

    # Constructing set of destination points to obtain a "birds eye view" (i.e. top-down view) of the image,
    # again specifying points in the top-left, top-right, bottom-right, and bottom-left order
    # We get the dimensions of the new image based on the width and height calculated
    dest = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    # making it float32 as getPerspectiveTransform requires it

    # compute the perspective transform matrix and then apply it
    transformation_matrix = cv2.getPerspectiveTransform(rect, dest)
    warped = cv2.warpPerspective(image, transformation_matrix, (maxWidth, maxHeight))

    return warped

#file='scan_4.jpg'
file = args["image"]
image = cv2.imread(file)
original = image.copy()
(h,w) = original.shape[:2]
# Calculating aspect ratio, to scale back the image if required when displaying the output
aspect_ratio =  h / 500.0
dim = (int(w * aspect_ratio), h)
#In order to speedup image processing, as well as make our edge detection step more accurate resizing image
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#  perform Gaussian blurring to remove high frequency noise (aiding in contour detection), and performing Canny edge detection.
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray, 5, 100)

##print("Edge detection ")
##cv2.imshow("Blurred Gray", gray)
cv2.imshow("Edged", edged)
cv2.waitKey(1000)

# Finding contours
cnts_mat = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts_mat[1] # as we have cv2 version 3.4

#sorting the contours by area and keep only the largest ones. This allows us to only examine the largest of the contours, discarding the rest.
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
	# if our approximated contour has four points, then can assume that as our surface
    if len(approx) == 4:
        op_Cnt = approx
        break

##print("Finding contours of object")
cv2.drawContours(image, [op_Cnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(1000)

warped = four_point_perspective_transform(image, op_Cnt.reshape(4, 2))
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
##print("Applying perspective transform")
cv2.imshow("Scanned", warped)
cv2.waitKey(3000)

cv2.destroyAllWindows()
