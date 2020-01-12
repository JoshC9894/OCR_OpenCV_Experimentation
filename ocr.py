import cv2
import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np 
import matplotlib.pyplot as plt
import pytesseract
#import peakutils
#import plotly.plotly as py
#import plotly.graph_objs as go


# load the image and compute the ratio of the old height to the new height, clone it, and resize it
image = cv2.imread("original_images/IMG_1947.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
 
# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
 
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
# cv2.imshow("Edged", edged)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
 
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 0, 200), 2)

# apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold itto give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian") 
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imwrite("out.jpg", warped)
img = cv2.imread("out.jpg")


#grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#grayscaled = cv2.bitwise_not(grayscaled)
#
#grayscaled = cv2.GaussianBlur(grayscaled, (17,17), 0) # 17 gives the best Tesseract results
#
## Apply OTSU Binirization
#retval2, threshold2 = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
## Vertical Histogram of the Binirized Image
#img_row_sum = np.sum(threshold2,axis=1).tolist()
## histogram = np.histogram(img_row_sum)
## plt.hist(histogram)
## plt.show()
#
##plt.plot(histogram) # Create Graph
##plt.show() # Show Graph
#
#cv2.imwrite("wraped3.jpg", threshold2)
#img = cv2.imread("wraped3.jpg")
#
## OCR with Tesseract
## text = pytesseract.image_to_string(cv2.imread("wraped3.jpg"))
## print(text)
#boundingBoxes = pytesseract.image_to_boxes(cv2.imread("wraped3.jpg"))
#
# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use
data = pytesseract.image_to_data(img)
h, w, _ = img.shape
#print(data)
# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 3)

# show annotated image and wait for keypress
cv2.imwrite("bounding.jpg", img)

