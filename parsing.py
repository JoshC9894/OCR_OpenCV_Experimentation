import cv2
import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import pytesseract
import re

# load the image and compute the ratio of the old height to the new height, clone it, and resize it
image = cv2.imread("IMG_7821.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
cv2.imwrite("1-edge.jpg", edged)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
# cv2.imshow("Edged", edged)
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

print("STEP 2: Find contours of paper")
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	print(approx)
	# if our approximated contour has four points, then we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		print(approx)
		break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

print("STEP 3: Apply perspective transform")
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.imwrite("2-warped.jpg", warped)

print("STEP 4: Apply OTSU Binirization")
# Remove Colour
grayscaled = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
grayscaled = cv2.bitwise_not(grayscaled)

# Apply OTSU Binirazation
retval2, threshold2 = cv2.threshold(grayscaled, 75, 200, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
threshold2 = cv2.GaussianBlur(threshold2, (7, 7), 0)
cv2.imwrite("3-threshold.jpg", threshold2)

print("STEP 5: Finding bounding boxes")
# Draw Bounding Boxes
img = cv2.imread("3-threshold.jpg")
h, w, _ = img.shape
boxes = pytesseract.image_to_boxes(img, lang="eng")

lines = []

def hasLine(top, bottom):
	for point in lines:
		if top < point < bottom:
			return True
	return False

for b in boxes.splitlines():
	b = b.split(' ')

	top = int(b[2])
	bottom = int(b[4])
	mid = int(top + ((bottom - top)/2))

	if len(lines) == 0:
		lines.append(mid)
	else:
		if hasLine(top, bottom) == False:
			lines.append(mid)

print("STEP 6: Find lines of text")

def isTotal(text):
	whitelist = ["total", "amount", "payment", "balance"]
	blacklist = ["before", "sub"]
#    Check blacklisted words
	for word in blacklist:
		if word in text.lower():
			return False
#   Check whitelist words
	for word in whitelist:
		if word in text.lower():
			return True
	return False

result = "No Result"
for line in lines:

	top = line
	bottom = line
	for box in boxes.splitlines():
		box = box.split(' ')
		if int(box[2]) < line < int(box[4]):
			if top > int(box[2]):
				top = int(box[2])
			if bottom < int(box[4]):
				bottom = int(box[4])
	crop = img[h - bottom: h - top, 0: w]

	text = pytesseract.image_to_string(crop, lang="eng")
	noSpace = text.replace(" ", "")
#    print()
#    print(text)
	price = re.findall("\d+\.\d\d", noSpace)
	if len(price) != 0 and isTotal(text):
		result = price[0]

	top = h
	bottom = 0

print("========|| DONE ||========")
print("The total is " + str(result))
