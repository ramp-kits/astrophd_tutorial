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
            self.y_pred = np.reshape(y_pred, (len(y_pred), self.n_columns))
        elif y_true is not None:
            y_true = np.reshape(y_true, (len(y_true), self.n_columns))
            # self._init_from_pred_labels(y_true)
            self.y_pred = y_true
        elif n_samples is not None:
            self.y_pred = np.empty((n_samples, self.n_columns), dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        return cls(y_pred=predictions_list[0].y_pred)


# An object implementing the workflow
workflow = rw.workflows.ObjectDetector()


def iou_bitmap(y_true, y_pred, verbose=False):
    """
    Compute the IoU between two arrays

    If the arrays are probabilities (floats) instead of predictions (integers
    or booleans) they are automatically rounded to the nearest integer and
    converted to bool before the IoU is computed.

    Parameters
    ----------
    y_true : ndarray
        array of true labels
    y_pred : ndarray
        array of predicted labels
    verbose : bool (optional)
        print the intersection and union separately

    Returns
    -------
    float :
        the intersection over union (IoU) value scaled between 0.0 and 1.0

    """
    EPS = np.finfo(float).eps

    # Make sure each pixel was predicted e.g. turn probability into prediction
    if y_true.dtype in [np.float32, np.float64]:
        y_true = y_true.round().astype(bool)

    if y_pred.dtype in [np.float32, np.float64]:
        y_pred = y_pred.round().astype(bool)

    # Reshape to 1d
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Compute intersection and union
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true + y_pred)
    jac = (intersection + EPS) / (sum_ - intersection + EPS)

    if verbose:
        print('Intersection:', intersection)
        print('Union:', sum_ - intersection)

    return jac


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
        iou_list = [iou_bitmap(yt, yp)
                    for (yt, yp) in zip(y_true, y_pred)]
        iou = np.mean(iou_list)

        return iou


score_types = [
    DeblendingScore()
]


def get_cv(X, y):
    # for simplicity just cut data in 3
    # for each fold use one third as test set, the other two as training

    n_tot = len(X)
    n1 = n_tot // 3
    n2 = n1 * 2
#
    return [(np.r_[0:n2], np.r_[n2:n_tot]),
            # (np.r_[n1:n_tot], np.r_[0:n1]),
            # (np.r_[0:n1, n2:n_tot], np.r_[n1:n2])]
            ]


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


def save_y_pred(y_pred, data_path, output_path, suffix):
    """Save a prediction vector in file.

    The file is (typically) in
    submissions/<submission_name>/training_output/y_pred_<suffix>.npz or
    submissions/<submission_name>/training_output/fold_<i>/y_pred_<suffix>.npz.

    Parameters
    ----------
    y_pred : a prediction vector
        a vector of predictions to be saved
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    output_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.npz will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.npz, can be used in
        problem.save_y_pred to, e.g., save only test predictions

    """
    y_pred_f_name = os.path.join(output_path, 'y_pred_{}'.format(suffix))

    if y_pred.ndim > 2:
        y_pred = y_pred.reshape(len(y_pred), -1)

    # Make sure array is rounded
    y_pred = y_pred.round()
    # Convert to boolean to save space and enble high compression
    y_pred = y_pred.astype(bool)

    np.savez_compressed(y_pred_f_name, y_pred=y_pred)
