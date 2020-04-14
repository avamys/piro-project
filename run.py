import os
import cv2
import numpy as np
from scipy import ndimage
import math
import re
import sys


class Preprocessing:

    def __init__(self):
        pass

    def rotate_image(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 60, minLineLength=10, maxLineGap=5)

        angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)

            median_angle = np.median(angles)
            img_rotated = ndimage.rotate(img, median_angle)

            return img_rotated
        else:
            return img

    def zoom_image(self, img):
        img_conv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_conv, 170, 255, cv2.THRESH_BINARY)

        rows = thresh[np.any(thresh, axis=1), :]
        norm = rows[:, np.any(rows, axis=0)]

        return norm

    def normalize_image(self, img, kernel, size):
        sel_blur = cv2.GaussianBlur(img, kernel, 0)
        img_norm = cv2.resize(sel_blur, size, interpolation=cv2.INTER_LINEAR)

        if np.sum(img_norm[-3, :]) < np.sum(img_norm[3, :]):
            img_norm = cv2.flip(img_norm, -1)

        ret, thresh = cv2.threshold(img_norm, 127, 255, cv2.THRESH_BINARY)

        normalized = thresh[np.any(thresh==False, axis=1), :]
        return normalized

    def preprocess(self, img):
        img_rotated = self.rotate_image(img)
        img_zoomed = self.zoom_image(img_rotated)

        dim = int(img_zoomed.shape[0] / 25)
        if dim > 2 and dim % 2 == 0:
            dim -= 1
        elif dim <= 2:
            dim = 3

        img_normalized = self.normalize_image(img_zoomed, (dim, dim), (735, 500))

        return img_normalized


numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_images_from_folder(folder, amount):
    images = []
    files = sorted(os.listdir(folder), key=numericalSort)
    #for filename in files:
    for i in range(amount):
        print(files[i])
        img = cv2.imread(os.path.join(folder, files[i]))
        if img is not None:
            images.append(img)
    return images


def image_description(img, step, sensitivity):
    descr = []
    for i in range(img.shape[1] // step):
        col = img[:, i * step]
        pix_sum = (col > sensitivity).sum()
        descr.append(pix_sum)

    return descr


def compare(descriptions, heights):
    dist_container = []
    desc = np.asarray(descriptions)

    mins = []

    for i in range(len(descriptions)):
        dist = []
        row = desc[i, :]

        for k in range(desc.shape[0]):
            diffs = desc[k, ::-1]
            euclidean_distance = np.sum(np.abs(heights[k] - row - diffs))
            if euclidean_distance == 0:
                dist.append(1000)
            else:
                dist.append(euclidean_distance)

        dist_container.append(dist)
        mins.append(dist.index(min(dist)))
    return dist_container, mins


if __name__ == "__main__":
    images = load_images_from_folder(sys.argv[1], int(sys.argv[2]))  # daneA/setX
    prep = Preprocessing()
    descriptions = []
    heights = []

    for idx, img in enumerate(images):
        img = prep.preprocess(img)
        des = image_description(img, 5, 200)
        descriptions.append(des)
        heights.append(img.shape[0])
        cv2.imshow('image' + str(idx), img)

    print("Heights: ")
    print(heights)

    desc_t = np.asarray(descriptions)
    print("Descriptions:")
    print(desc_t)

    print("--------------------------------------------------")
    print("Similarity:")
    similarities, mins = compare(descriptions, heights)
    print(similarities)
    print("--------------------------------------------------")
    print("Best matches:")
    print(mins)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
