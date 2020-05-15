import numpy as np
import pandas as pd
import math
import os


def load_data(path, names=None):
    if os.path.exists(path):
        df = pd.read_csv(path, names=names)
        return df
    raise OSError('invalid path to data')


def encode_labels(labels):
    result = []
    s = set(labels)
    d = dict((k, v) for v, k in enumerate(s))
    for label in labels:
        res = d[label]
        result.append(res)
    return result


class KNearestNeighbors:

    def __init__(self):  # I know KNN meant to have no parameters but ... It's just my way ¯\_ツ_/¯
        self.metrics = {'euclidean': self._euclidean, 'manhattan': self._manhattan,
                        'hamming': self._hamming, 'minkowski': self._minkowski}
        self.features = []
        self.labels = []

    def fit(self, x, y) -> None:
        self.features = x
        self.labels = y

    def predict(self, x, neighbors=5, metrics='euclidean') -> list:
        if neighbors >= len(x):
            raise Exception('Number of neighbors is greater or equal to total number of samples')
        result = []
        for vals in x:
            distances = []
            counter = [0]*len(set(self.labels))
            for point, label in zip(self.features, self.labels):
                dist = self.metrics[metrics](vals, point)
                distances.append((dist, label))
            distances = sorted(distances, key=lambda v: v[0])[:neighbors]
            for _, i in distances:
                counter[i] += 1
            res = np.argmax(counter)
            result.append(res)
        return result

    @staticmethod
    def evaluate(y, y_hat) -> float:
        counter = 0
        for p, q in zip(y, y_hat):
            counter += 1 if p == q else 0
        return counter/len(y)

    @staticmethod
    def _euclidean(x1, x2) -> float:
        dst = sum((v1-v2)**2 for v1, v2 in zip(x1, x2))
        return math.sqrt(dst)

    @staticmethod
    def _manhattan(x1, x2) -> float:
        dst = sum(abs(v1-v2) for v1, v2 in zip(x1, x2))
        return dst

    @staticmethod
    def _hamming(x1, x2) -> float:
        counter = 0
        for v1, v2 in zip(x1, x2):
            counter += 1 if v1 != v2 else 0
        return counter

    @staticmethod
    def _minkowski(x1, x2) -> float:
        n = len(x1)
        dst = sum((v1-v2)**n for v1, v2 in zip(x1, x2))
        return dst**(1/n)
