import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


def load_img():
    start_time = time.time()
    # folder_name = 'Capsule images'
    folder_name = 'Single Softgel/Softgel'
    img = cv2.imread(f'{folder_name}/Oval1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    visual_img = img.copy()
    # Find Canny edges
    edged = cv2.Canny(gray, 20, 200)  # 20
    plt.imshow(edged)
    plt.show()

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))
    cv2.drawContours(visual_img, contours, -1, (0, 255, 0), 2)

    plt.imshow(visual_img, cmap='gray')
    plt.show()

    extract_object(gray, contours, start_time)


def extract_object(gray, contours, start_time):
    label = []
    object_list = []
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    for i in range(len(contours)):
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, i, color=255, thickness=-1)
        index_pixel = np.where(mask == 255)

        result = np.zeros((gray.shape[0], gray.shape[1]))
        result[index_pixel[0], index_pixel[1]] = gray[index_pixel[0], index_pixel[1]]
        object_list.append((index_pixel[0], index_pixel[1]))

        idx_txt_col = np.min(index_pixel[1])

        status = anomaly_detection(result, idx_txt_col, i)
        label.append(status)
        if status == 'Accept':
            color_img[index_pixel[0], index_pixel[1], 1] = 255
        elif status == 'Reject':
            color_img[index_pixel[0], index_pixel[1], 0] = 255
    # plt.imshow(color_img, cmap='gray')
    # plt.show()
    print(label)
    end_time = time.time()
    print('time is:', (end_time - start_time))


def anomaly_detection(image, txt_idx, num):
    status = 'Accept'
    image = image.astype(dtype='uint8')
    canny = cv2.Canny(image, 120, 200)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


    img_copy = np.copy(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    for i, c in enumerate(contours):
        areaContours = cv2.contourArea(c)
        print(areaContours)

        # if areaContours > 100 and areaContours < 1000: # Red Capsule
        if 1000 < areaContours < 2000: # Transparent Oval
            # print(f'area {i}:', areaContours)
            status = 'Reject'
            cv2.drawContours(img_copy, contours, i, (0, 0, 255), -1)
    if status == 'Reject':
        cv2.putText(img=img_copy, text=status, org=(txt_idx, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(30, 10, 200), thickness=2)
    else:
        cv2.putText(img=img_copy, text=status, org=(txt_idx, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(10, 200, 30), thickness=2)
    cv2.imwrite(f'Results/Labeling Capsules/Oval {num + 1:02}.png', img_copy)
    plt.imshow(img_copy,cmap='gray')
    plt.show()
    return status


if __name__ == '__main__':
    load_img()
