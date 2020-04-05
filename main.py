import os
import cv2
import numpy as np
from preprocessing import Preprocessing
from model import Model


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
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


if __name__ == "__main__":
    images = load_images_from_folder('daneA/set0')
    prep = Preprocessing()
    mod = Model()
    descriptions = []

    for idx, img in enumerate(images):
        img = prep.preprocess(img)
        des = image_description(img, 15, 200)
        descriptions.append(des)
        cv2.imshow('image' + str(idx), img)

    desc_t = np.asarray(descriptions)
    print(desc_t.shape)
    print(desc_t)

    print("--------------------------------------------------")
    similarities, mins = mod.compare(descriptions)
    print(similarities)
    print("--------------------------------------------------")
    print(mins)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
