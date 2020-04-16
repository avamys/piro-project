import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr
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

            if img_rotated.shape[0] > img_rotated.shape[1]:
                img_rotated = ndimage.rotate(img_rotated, 90)

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

        normalized = thresh[np.any(thresh == False, axis=1), :]
        return normalized

    def cut_sides(self, img):
        cut = img[:450, 50:-50]
        ret, thresh = cv2.threshold(cut, 127, 255, cv2.THRESH_BINARY)
        norm = cv2.resize(thresh, (735, thresh.shape[0]), interpolation=cv2.INTER_LINEAR)
        return norm

    def preprocess(self, img):
        img_rotated = self.rotate_image(img)
        img_zoomed = self.zoom_image(img_rotated)

        dim = int(img_zoomed.shape[0] / 25)
        if dim > 2 and dim % 2 == 0:
            dim -= 1
        elif dim <= 2:
            dim = 3

        img_normalized = self.normalize_image(img_zoomed, (dim, dim), (735, 500))

        cut = self.cut_sides(img_normalized)
        return cut


numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_images_from_folder(folder, amount):
    images = []
    files = sorted(os.listdir(folder), key=numerical_sort)
    for i in range(amount):
        # print(files[i])
        img = cv2.imread(os.path.join(folder, files[i]))
        if img is not None:
            images.append(img)
    return images


def load_corrects(folder):
    correct = []
    f = open(folder + '/correct.txt', 'r')
    for x in f:
        correct.append(int(x))
    return correct


def image_description(img, step, sensitivity):
    descr = []
    for i in range(img.shape[1] // step):
        col = img[:, i * step]
        pix_sum = (col > sensitivity).sum()
        descr.append(pix_sum)

    return descr


def compare2(descriptions):
    dist_container = []
    desc = np.asarray(descriptions)

    mins = []

    for i in range(len(descriptions)):
        dist = []
        row = desc[i, :]

        for k in range(desc.shape[0]):
            diffs = desc[k, ::-1]
            corr = pearsonr(row, diffs)
            if i == k:
                dist.append((1000, 1000))
            else:
                dist.append(corr)

        dist_container.append(dist)
        mins.append(dist.index(min(dist)))
    return dist_container, mins


def make_ranking(dists):
    for i, dist in enumerate(dists):
        matching = sorted(range(len(dist)), key=lambda k: dist[k])
        matching.remove(i)
        print(*matching, sep=' ')


if __name__ == "__main__":
    images = load_images_from_folder(sys.argv[1], int(sys.argv[2]))
    prep = Preprocessing()
    descriptions = []

    for idx, img in enumerate(images):
        img = prep.preprocess(img)
        des = image_description(img, 5, 200)
        descriptions.append(des)
        # cv2.imshow('image' + str(idx), img)

    desc_t = np.asarray(descriptions)

    similarities, mins = compare2(descriptions)
    print("Best matches:")
    print(mins)

    corrects = load_corrects(sys.argv[1])
    matches = [i for i, j in zip(mins, corrects) if i == j]
    accuracy = len(matches) / len(mins)
    print("Accuracy: ", accuracy)

    print("--------------------------------------------------")
    make_ranking(similarities)

    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()
