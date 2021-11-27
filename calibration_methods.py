import numpy as np
import cvxpy as cp
from sklearn.isotonic import IsotonicRegression

from utils import determine_edges
from utils import bin_confidences_and_accuracies

# for BetaCalibration
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

import warnings


def get_confidences(scores_cal, scores_test, ground_truth_cal, nbins, score_min=-1, score_max=1):
    bin_edges, binning_indices = determine_edges(scores_cal, nbins, indicesQ=True, score_min=score_min,
                                                 score_max=score_max)

    print('Statistics Calibration')
    print('Total number of scores: %d' % len(scores_cal))
    for i in range(len(bin_edges) - 1):  # we use bin_edges since we may have not been able to form nbins
        print('Mass in bin %d: %d' % (i, sum(binning_indices == i)))

    weights, bin_confidence, _ = bin_confidences_and_accuracies(scores_cal, ground_truth_cal, bin_edges,
                                                                binning_indices)

    confidences_cal = bin_confidence[np.maximum(np.digitize(scores_cal, bin_edges, right=True) - 1, 0)]
    confidences_test = bin_confidence[np.maximum(np.digitize(scores_test, bin_edges, right=True) - 1, 0)]

    return confidences_cal, confidences_test


def get_confidences_isotonic_regression(scores_cal, scores_test, ground_truth_cal, score_min=-1, score_max=1):
    iso_reg = IsotonicRegression(
        y_min=score_min,
        y_max=score_max,
        out_of_bounds='clip').fit(scores_cal, ground_truth_cal)
    confidences_cal = iso_reg.predict(scores_cal)
    if len(scores_test) > 0:
        confidences_test = iso_reg.predict(scores_test)
    else:
        confidences_test = scores_test.copy()
    return confidences_cal, confidences_test


class IsotonicCalibration():
    def __init__(self, scores, ground_truth, score_min=-1, score_max=1):
        self.score_min = score_min
        self.score_max = score_max
        self._obj = IsotonicRegression(
            y_min=self.score_min,
            y_max=self.score_max,
            out_of_bounds='clip').fit(scores, ground_truth)

    def predict(self, scores):
        return self._obj.predict(scores)


class BinningCalibration():
    def __init__(self, scores, ground_truth, score_min=-1, score_max=1, nbins=10):
        self.bin_edges, self.binning_indices = determine_edges(
            scores, nbins, indicesQ=True, score_min=score_min, score_max=score_max)
        self.weights, self.bin_confidence, _ = bin_confidences_and_accuracies(
            scores, ground_truth, self.bin_edges, self.binning_indices)

    def predict(self, scores):
        return self.bin_confidence[np.maximum(np.digitize(scores, self.bin_edges, right=True) - 1, 0)]


def normalize(score, score_min=-1, score_max=1):
    return (score-score_min)/(score_max-score_min)


# The code below was adapted from https://github.com/betacal/betacal.github.io the original repo for the paper
# Beta regression model with three parameters introduced in
#     Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
#     and easily implemented improvement on logistic calibration for binary
#     classifiers. AISTATS 2017.

def _beta_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1-eps)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(x)
    x[:, 1] *= -1

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]

    if coefs[0] < 0:
        x = x[:, 1].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = 0
        b = coefs[0]
    elif coefs[1] < 0:
        x = x[:, 0].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = coefs[0]
        b = 0
    else:
        a = coefs[0]
        b = coefs[1]
    inter = lr.intercept_[0]

    m = minimize_scalar(lambda mh: np.abs(b*np.log(1.-mh)-a*np.log(mh)-inter),
                        bounds=[0, 1], method='Bounded').x
    map = [a, b, m]
    return map, lr


class BetaCalibration(BaseEstimator, RegressorMixin):
    """Beta regression model with three parameters introduced in
    Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
    and easily implemented improvement on logistic calibration for binary
    classifiers. AISTATS 2017.
    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m]
    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """

    def __init__(self, scores, ground_truth, score_min=-1, score_max=1):
        self.score_min = score_min
        self.score_max = score_max
        X = column_or_1d(normalize(scores, score_min=score_min, score_max=score_max))
        y = column_or_1d(ground_truth)
        X, y = indexable(X, y)
        self.map_, self.lr_ = _beta_calibration(X, y, )

    def predict(self, S):
        """Predict new values.
        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.
        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(normalize(S, score_min=self.score_min, score_max=self.score_max)).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1 - eps)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        return self.lr_.predict_proba(x)[:, 1]