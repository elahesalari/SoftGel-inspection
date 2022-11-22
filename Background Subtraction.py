import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance, ImageFilter


class ExtractObject:
    def __init__(self, sharp, contrast, threshold_value, path_read, path_write):
        self.sharp = sharp
        self.contrast = contrast
        self.thresh_val = threshold_value
        self.path_read = path_read
        self.path_write = path_write

    def remove_background(self, filename):
        img = Image.open(os.path.join(self.path_read, filename))
        img_array = np.array(img)

        enhancer = ImageEnhance.Sharpness(img)
        res = enhancer.enhance(self.sharp)  # defect: 6 , capsule: 5

        enhancer = ImageEnhance.Contrast(res)
        res = enhancer.enhance(self.contrast)  # defect: 6 , capsule: 2

        res = np.array(res)
        image = cv2.fastNlMeansDenoisingColored(res, None, 20, 20, 7, 21)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Binarize
        thresh, binary_image = cv2.threshold(gray, self.thresh_val, 255,
                                             cv2.THRESH_BINARY)  # defect: 225 , capsule: 160

        # Create mask
        mask = np.zeros(image.shape, np.uint8)
        mask.fill(255)

        # get most significant contour
        contours_sign, hierachy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print('The count of contours:', len(contours_sign))

        cv2.fillConvexPoly(mask, contours_sign[1], (0, 0, 0))
        th, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)

        new_image = cv2.bitwise_and(img_array, mask)
        # plt.imshow(new_image)
        # plt.show()

        return new_image, filename

    def save_object(self, image, filename):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(image)
        rgba = [b, g, r, thresh]
        result = cv2.merge(rgba, 4)
        cv2.imwrite(f'{self.path_write}/{filename[:-4]}.png', result)
        # plt.imshow(result)
        # plt.show()
        return result




if __name__ == '__main__':
    path_read = 'Camera images/Multi gel'
    path_write = 'Single Softgel/Softgel'

    for filename in os.listdir(path_read):
        print(filename)
        if filename.startswith('Oval'):
            sharp = 6
            contrast = 6
            threshold = 225

        if filename.startswith('Capsule'):
            sharp = 5
            contrast = 2
            threshold = 160

        ex_obj = ExtractObject(sharp=sharp, contrast=contrast, threshold_value=threshold, path_read=path_read,
                               path_write=path_write)
        image, filename = ex_obj.remove_background(filename=filename)
        result = ex_obj.save_object(image, filename)

