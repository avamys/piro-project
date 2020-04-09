import numpy as np


class Model:
    def __init__(self):
        pass


    def compare(self, descriptions):
        dist_container = []
        desc = np.asarray(descriptions)

        # TEST
        mins = []

        for i in range(len(descriptions)):
            dist = []
            row = np.abs(np.diff(desc[i, :]))

            for k in range(desc.shape[0]):
                diffs = np.abs(np.diff(desc[k, ::-1]))
                euclidean_distance = np.abs(np.sum(row - diffs))
                if euclidean_distance == 0:
                    dist.append(1000)
                else:
                    dist.append(euclidean_distance)

            dist_container.append(dist)
            mins.append(dist.index(min(dist)))
        return dist_container, mins

