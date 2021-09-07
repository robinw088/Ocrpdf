import cv2
import numpy as np
import pandas as pd
import pytesseract
from os import path
from collections import defaultdict


class ProcessScan:

    def __init__(self, file, ):
        self.file = file
        self.img = cv2.imread(file, 0)
        self.shape = self.img.shape
        self.column = 1
        self.data_dict = defaultdict(list)
        self.total_row = 1

    def ocr_pdf(self):

        # thresholding the image to a binary image
        thresh, img_bin = cv2.threshold(self.img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # inverting the image
        img_bin = 255 - img_bin
        cv2.imwrite('./data/cv_inverted.png', img_bin)

        # countcol(width) of kernel as 100th of total width
        kernel_len = np.array(self.img).shape[1] // 100
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

        '''Detect contours for following box detection '''

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        ctr_filtered = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if 'RM' in self.file:
                self.column = 6
                contours_conditions = 10 < w < self.img.shape[1] / 2 and 10 < h < self.shape[0]
            else:
                self.column = 3
                contours_conditions = 0 < w < self.img.shape[1] and self.img.shape[0] / 15 < h < self.img.shape[
                    0] * 0.81

            if contours_conditions:
                ctr_filtered.append(c)

        ctr_arr = np.ones(shape=(len(ctr_filtered), 4), dtype=int)

        self.total_row = int(len(ctr_arr) / self.column)

        for idx, c in enumerate(ctr_filtered):
            x, y, w, h = cv2.boundingRect(c)
            ctr_arr[idx] = [x, y, w, h]

        ctr_arr = ctr_arr[np.argsort(ctr_arr[:, 0])]

        if 'RM' not in self.file:
            ctr_arr[-self.total_row:, [1, 3]] = ctr_arr[:self.total_row, [1, 3]]

        data_dict = defaultdict(list)

        for i in range(self.column):
            column = ctr_arr[:self.total_row]
            column = column[np.argsort(column[:, 1])]
            for row in column:
                x, y, w, h = row
                image_crop = self.img[y:y + h, x:x + w]
                cv2.imshow('img', image_crop)
                cv2.waitKey()
                data_text = pytesseract.image_to_string(image_crop).strip()
                data_dict[i].append(data_text)
            ctr_arr = ctr_arr[self.total_row:]
        df_data = pd.DataFrame.from_dict(data_dict)
        df_data.to_csv(path.join(path.split(self.file)[0], 'output.csv'), mode='a')
