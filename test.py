import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from collections import defaultdict

FP_COLUMNS=3
# read your file
file = r'./data_fp/20210417 fp_scan.jpg'
img = cv2.imread(file, 0)
shape = img.shape
print(shape)
# thresholding the image to a binary image
thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# inverting the image
img_bin = 255 - img_bin
cv2.imwrite('./data_fp/cv_inverted.png', img_bin)
plt.imshow(img_bin)


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
cv2.imwrite("./data_fp/img_vh.jpg", img_vh)

im = cv2.imread("./data_fp/img_vh.jpg")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

# Detect contours for following box detection

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

blank_image = np.ones((shape[0], shape[1], 3), np.uint8)


df_box = pd.DataFrame(columns=['x', 'y', 'w', 'h'])
# Get position (x,y), width and height for every contour and show the contour on image
for index, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    if 0 < w < img.shape[1] and img.shape[0]/15 < h < img.shape[0]*0.81:
        df_box.loc[index] = [x, y, w, h]

df_box = df_box.sort_values(by=['x','y'], ignore_index=True)
print(df_box)
total_row = int(len(df_box) / FP_COLUMNS)
df_box.iloc[-total_row:, [1, 3]] = df_box.iloc[:total_row, [1, 3]]

data_dict = defaultdict(list)

for i in range(FP_COLUMNS):
    df_box_column = df_box.iloc[:total_row, :].sort_values(by='y', )
    for idx, row in df_box_column.iterrows():
        x, y, w, h = row
        image_crop = img[y:y + h, x:x + w]
        cv2.imshow('Display', image_crop)
        cv2.waitKey()
        data_text = pytesseract.image_to_string(image_crop).strip()
        data_dict[i].append(data_text)
    df_box = df_box.iloc[total_row:, :]

df_data = pd.DataFrame.from_dict(data_dict)
df_data.to_csv('./data/output.csv')
print(df_data)
