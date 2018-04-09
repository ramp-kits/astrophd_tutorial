import os

import numpy as np

import rampwf as rw
from rampwf.score_types.base import BaseScoreType


problem_title = 'Astro PhD tutorial - galaxy deblending'

# A type (class) which will be used to create wrapper objects for y_pred
MultiClassPredictions = rw.prediction_types.make_multiclass(label_names=[0, 1])


class Predictions(MultiClassPredictions):
    # Each pixel has to be classified independandly
    n_columns = 128 * 128

    def __init__(self, y_pred=None, y_true=None, n_samples=None):
        if y_pred is not None:
            size = len(y_pred)
            self.y_pred = np.reshape(y_pred, (size, self.n_columns))
        elif y_true is not None:
            size = len(y_true)
            y_true = np.reshape(y_true, (size, self.n_columns))
            # self._init_from_pred_labels(y_true)
            self.y_pred = y_true
        elif n_samples is not None:
            self.y_pred = np.empty((n_samples, self.n_columns), dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()


# An object implementing the workflow
workflow = rw.workflows.ObjectDetector()


def iou(y_true, y_pred):
    EPS = np.finfo(float).eps
    y_true_b = y_true.astype(bool)
    y_pred_b = y_pred.astype(bool)

    intersection = np.sum(y_true_b & y_pred_b, axis=-1)
    sum_ = np.sum(y_true_b | y_pred_b, axis=-1)
    jaccard = (intersection + EPS) / (sum_ + EPS)

    return jaccard


class DeblendingScore(BaseScoreType):
    """
    """
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='IoU', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        iou_score = iou(y_true, y_pred)
        return np.mean(iou_score)


score_types = [
    DeblendingScore()
]


def get_cv(X, y):
    # for simplicity just cut data in 3
    # for each fold use one third as test set, the other two as training

    n_tot = len(X)
    n1 = n_tot // 3
    n2 = n1 * 2

    return [(np.r_[0:n2], np.r_[n2:n_tot]),
            (np.r_[n1:n_tot], np.r_[0:n1]),
            (np.r_[0:n1, n2:n_tot], np.r_[n1:n2])]


def _read_data(path, typ):
    """
    Read and process data and labels.

    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'train', 'test'}

    Returns
    -------
    X, y data

    """
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        suffix = '_mini'
    else:
        suffix = ''

    try:
        data_path = os.path.join(path, 'data',
                                 'data_{0}{1}.npy'.format(typ, suffix))
        # X = np.load(data_path, mmap_mode='r')
        X = np.load(data_path).squeeze()

        labels_path = os.path.join(path, 'data',
                                   'labels_{0}{1}.npy'.format(typ, suffix))
        # y = np.load(labels_path, mmap_mode='r')
        y = np.load(labels_path).squeeze()
    except IOError:
        raise IOError("'data/data_{0}.npy' and 'data/labels_{0}.csv' are not "
                      "found. Ensure you ran 'python download_data.py' to "
                      "obtain the train/test data".format(typ))

    return X, y


def get_test_data(path='.'):
    return _read_data(path, 'test')


def get_train_data(path='.'):
    return _read_data(path, 'train')
