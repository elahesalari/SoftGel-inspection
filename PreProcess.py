''' This file is the first file:
    that get one image from multiple soft gel and apply contour on this, separate each soft gel,
    finally, save each single soft gel in separate files.
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
from PIL import Image, ImageEnhance, ImageFilter


class Preprocess:
    def __int__(self, path_read, path_write):
        self.path_read = path_read
        self.path_write = path_write

    def load_img(self):
        start_time = time.time()
        for filename in os.listdir(self.path_read):
            img = cv2.imread(os.path.join(self.path_read, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            visual_img = img.copy()
            # Find Canny edges
            edged = cv2.Canny(gray, 30, 200)

            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print("Number of Contours found = " + str(len(contours)))
            cv2.drawContours(visual_img, contours, -1, (0, 255, 0), 3)

            self.extract_object(img, gray, contours, start_time,filename)

    def extract_object(self, img, gray, contours, start_time, filename):
        count = len([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 2000])
        print(count)
        cap_nums = 0
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 2000:
                cap_nums += 1
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, contours, i, color=255, thickness=-1)
                index_pixel = np.where(mask == 255)
                min_row, max_row = np.min(index_pixel[0]), np.max(index_pixel[0])
                min_col, max_col = np.min(index_pixel[1]), np.max(index_pixel[1])
                color_cutted = img[min_row:max_row, min_col:max_col]

                object = np.zeros_like(gray)
                object[index_pixel[0], index_pixel[1]] = gray[index_pixel[0], index_pixel[1]]
                gray_cutted = object[min_row:max_row, min_col:max_col]

                _, thresh = cv2.threshold(gray_cutted, 1, 255, cv2.THRESH_BINARY)
                r, g, b = cv2.split(color_cutted)
                rgba = [r, g, b, thresh]
                result = cv2.merge(rgba, 4)


                cv2.imwrite(f'{self.path_write}/{filename+str(cap_nums)}.png', result)
        end_time = time.time()
        print('time is:', (end_time - start_time))



if __name__ == '__main__':
    path_read = 'Images by camera'
    path_write = 'Separated_Softgels'
    pr = Preprocess()
    pr.path_read = path_read
    pr.path_write = path_write
    pr.load_img()
