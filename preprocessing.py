import cv2
import numpy as np
import math
from scipy import ndimage

class Preprocessing:

    def __init__(self):
        pass


    def rotate_image(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=10, maxLineGap=5)

        angles = []

        for x1, y1, x2, y2 in lines[0]:
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = np.median(angles)
        img_rotated = ndimage.rotate(img, median_angle)

        return img_rotated


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

        return img_norm


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