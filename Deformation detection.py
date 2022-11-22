import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance, ImageFilter
from scipy import signal


class Deformation:
    def __init__(self, w):
        self.W = w

    def read_truth(self, path, filename):
        truth = cv2.imread(os.path.join(path, filename))
        return truth

    def read_samples(self, path, filename):
        sample = cv2.imread(os.path.join(path, filename))
        return sample

    def process(self, object):
        mask, indecis, start_point = self.get_edges(object)
        truth_alpha = self.calc_degree(mask, indecis, start_point)
        shifted_signal = self.shift(truth_alpha)
        smooth = self.smooth(shifted_signal)
        return smooth

    def get_edges(self, image):
        edged = cv2.Canny(image, 100, 200)  # 20
        mask = np.zeros((image.shape[0], image.shape[1]))

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print("Number of Contours found = " + str(len(contours)))
        cv2.drawContours(mask, contours, -1, 255, 1)
        img_dilation = cv2.dilate(mask, (5, 5), iterations=1)
        img_erosion = cv2.erode(img_dilation, (3, 3), iterations=1)
        plt.imshow(img_erosion, cmap='gray')
        plt.show()

        indecis = np.where(img_erosion == 255)
        self.count_of_white_pixels = len(indecis[0])
        # print('count of white pixel:', self.count_of_white_pixels)
        max_point_y = np.max(indecis[1])
        max_point_x = indecis[0][indecis[1] == max_point_y][0]
        start_point = (max_point_x, max_point_y)
        return mask, indecis, start_point

    def calc_degree(self, binary_img, indecis, start_point):
        next_point = (0, 0)
        current_point = start_point
        before_point = (0, 0)
        alpha = []
        print('start point:', start_point)
        while (start_point != next_point):
            north = (current_point[0] - 1, current_point[1])
            north_east = (current_point[0] - 1, current_point[1] + 1)
            east = (current_point[0], current_point[1] + 1)
            south_east = (current_point[0] + 1, current_point[1] + 1)
            south = (current_point[0] + 1, current_point[1])
            south_west = (current_point[0] + 1, current_point[1] - 1)
            west = (current_point[0], current_point[1] - 1)
            north_west = (current_point[0] - 1, current_point[1] - 1)
            print('next point:', next_point)
            # North
            if binary_img[north[0], north[1]] == 255 and north != before_point:
                alpha.append(0)
                next_point = north
                before_point = current_point
                current_point = next_point
            # North-East
            elif binary_img[north_east[0], north_east[1]] == 255 and north_east != before_point:
                alpha.append(45)
                next_point = north_east
                before_point = current_point
                current_point = next_point
            # North-West
            elif binary_img[north_west[0], north_west[1]] == 255 and north_west != before_point:
                alpha.append(-45)
                next_point = north_west
                before_point = current_point
                current_point = next_point
            # East
            elif binary_img[east[0], east[1]] == 255 and east != before_point:
                alpha.append(90)
                next_point = east
                before_point = current_point
                current_point = next_point
            # West
            elif binary_img[west[0], west[1]] == 255 and west != before_point:
                alpha.append(-90)
                next_point = west
                before_point = current_point
                current_point = next_point
            # South
            elif binary_img[south[0], south[1]] == 255 and south != before_point:
                alpha.append(180)
                next_point = south
                before_point = current_point
                current_point = next_point
            # South-East
            elif binary_img[south_east[0], south_east[1]] == 255 and south_east != before_point:
                alpha.append(135)
                next_point = south_east
                before_point = current_point
                current_point = next_point
            # South-West
            elif binary_img[south_west[0], south_west[1]] == 255 and south_west != before_point:
                alpha.append(-135)
                next_point = south_west
                before_point = current_point
                current_point = next_point
        print('The length of alpha:', len(alpha))
        print(alpha)
        return alpha

    def smooth(self, angles):
        box = np.ones(self.W) / self.W
        smoothing = np.convolve(angles, box, mode='same')
        return smoothing

    def shift(self,signal):
        shifted = np.roll(signal, 15)
        signal_norm = signal - shifted
        return signal_norm

    def plot_signals(self, truth_signal, sample_signal, filename):
        # correlation_signals = signal.correlate(truth_signal, other_signal, mode='same')
        # delay = np.argmax(correlation_signals) - int(len(correlation_signals) / 2)
        # print('Dely:', delay)
        # plt.plot(correlation_signals, marker='.', color='blue', markerfacecolor='blue',
        #          label=f'Correlation between truth and {filename} signals')
        # difference = truth_signal - sample_signal
        plt.plot(truth_signal, color='blue', label='Truth Signal')
        plt.plot(sample_signal, color='red', label=f'{filename} Signal')
        plt.legend(loc='upper left')
        plt.show()


if __name__ == '__main__':
    w = 20
    df = Deformation(w)
    path_truth = 'Separated_Softgels/Truth Samples'
    path_samples = 'Separated_Softgels/Samples'
    softgel_names = ['Large_Ovals', 'Red_Capsules', 'Rounds', 'Small_Ovals', 'Y_Capsules']
    for sf in softgel_names:
        for filename in os.listdir(path_truth):
            if filename.startswith(sf):
                truth_img = df.read_truth(path_truth, filename)
                alpha_hat_truth = df.process(truth_img)
                for name_smp in os.listdir(path_samples):
                    if name_smp.startswith(sf):
                        sample_img = df.read_samples(path_samples, name_smp)
                        alpha_hat_sample = df.process(sample_img)
                        df.plot_signals(alpha_hat_truth, alpha_hat_sample, name_smp[:-4])




