import cv2
import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import pytesseract
import re

# load the image and compute the ratio of the old height to the new height, clone it, and resize it
image = cv2.imread("Receipts/input1.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
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

print("STEP 3: Apply perspective transform")
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.imwrite("output_images/2-warped.jpg", warped)

print("STEP 4: Apply OTSU Binirization")
# Remove Colour
grayscaled = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
grayscaled = cv2.bitwise_not(grayscaled)

# Apply OTSU Binirazation
retval2, threshold2 = cv2.threshold(grayscaled, 75, 200, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
threshold2 = cv2.GaussianBlur(threshold2, (7, 7), 0)
cv2.imwrite("output_images/3-threshold.jpg", threshold2)

print("STEP 4: Apply Dilation")
blur = cv2.GaussianBlur(threshold2, (9, 9), 0)
dilate = cv2.dilate(blur, np.ones((2, 150)))
cv2.imwrite("output_images/4-dilate.jpg", dilate)

print("STEP 5: Apply Erosion")
erosion = cv2.erode(dilate,np.ones((2, 75)))
cv2.imwrite("output_images/5-erode.jpg", erosion)

## Find the contours
print("STEP 6: Find region contours")
image,contours,hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,255,0),2)

print("STEP 6: Done")
cv2.imwrite("output_images/6-output.jpg", warped)
