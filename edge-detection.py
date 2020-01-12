import cv2
import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import pytesseract
import re

# load the image and compute the ratio of the old height to the new height, clone it, and resize it
image = cv2.imread("Receipts/IMG_2848.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray,cv2.CV_64F)
#gray = cv2.GaussianBlur(gray, (3, 3), 0)
#gray = cv2.dilate(gray, np.ones((25, 5)))
gray = cv2.erode(gray, np.ones((21, 21)))
edged = cv2.Canny(gray, 75, 200)
cv2.imwrite("output_images/0-edge.jpg", edged)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

#print("STEP 2: Find contours of paper")
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
# loop over the contours
for c in cnts:
    # approximate the contour
    epsilon = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.08 * epsilon, True)
    # if our approximated contour has four points, then we can assume that we have found our screen - This needs improving as the
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imwrite("output_images/1-contours.jpg", image)
