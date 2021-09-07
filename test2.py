import cv2
from collections import defaultdict
import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

RM_COLUMNS = 5
# read your file
file = r'./data/20210418 RM-4.JPG'
img = cv2.imread(file, 0)
shape = img.shape

# thresholding the image to a binary image
thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# inverting the image
img_bin = 255 - img_bin
cv2.imwrite('./data/cv_inverted.png', img_bin)

# countcol(width) of kernel as 100th of total width
kernel_len = np.array(img).shape[1] // 100
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

img_vh = cv2.addWeighted(vertical_lines, 2, horizontal_lines, 2, 0.0)
# Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("./data/img_vh.jpg", img_vh)

im = cv2.imread("./data/img_vh.jpg")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

# Detect contours for following box detection

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(contours[0])
contours_sorted = []
x, y, w, h = cv2.boundingRect(contours[0])
print(x,y,w,h)
cv2.waitKey()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if 10 < w < img.shape[1] / 2 and 10 < h < img.shape[0]:
        contours_sorted.append(c)

ctr_arr = np.ones(shape=(len(contours_sorted), 4), dtype=int)

total_row = int(len(ctr_arr) / RM_COLUMNS)

for idx, c in enumerate(contours_sorted):
    x, y, w, h = cv2.boundingRect(c)
    ctr_arr[idx] = [x, y, w, h]

ctr_arr = ctr_arr[np.argsort(ctr_arr[:, 0])]

print(ctr_arr)

print(ctr_arr[:8, [1,3]])

data_dict = defaultdict(list)

for i in range(RM_COLUMNS):
    column = ctr_arr[:total_row]
    column = column[np.argsort(column[:, 1])]
    for row in column:
        x, y, w, h = row
        image_crop = img[y:y + h, x:x + w]
        cv2.imshow('img',image_crop)
        cv2.waitKey()
        data_text = pytesseract.image_to_string(image_crop).strip()
        data_dict[i].append(data_text)
    ctr_arr = ctr_arr[total_row:]

df_data = pd.DataFrame.from_dict(data_dict)
df_data.to_csv('./data/output.csv')
print(df_data)
