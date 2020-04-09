import os
import cv2
import numpy as np
from preprocessing import Preprocessing
from model import Model
import re

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder), key=numericalSort):
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def image_description(img, samples, sensitivity):
    descr = []
    for i in range(img.shape[1] // samples):
        col = img[:, i * samples]
        pix_sum = (col > sensitivity).sum()
        descr.append(pix_sum)

    return descr


def describe2(cont):
    perimeter = 0
    for c in cont:
        perimeter += cv2.arcLength(c, True)
    return perimeter


if __name__ == "__main__":
    images = load_images_from_folder('daneA/set4')
    prep = Preprocessing()
    mod = Model()
    descriptions = []

    for idx, img in enumerate(images):
        img = prep.preprocess(img)
        des = image_description(img, 5, 200)
        descriptions.append(des)
        cv2.imshow('image' + str(idx), img)

    desc_t = np.asarray(descriptions)
    print("Descriptions:")
    print(desc_t)

    print("--------------------------------------------------")
    print("Similarity:")
    similarities, mins = mod.compare(descriptions)
    print(similarities)
    print("--------------------------------------------------")
    print("Best matches:")
    print(mins)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
