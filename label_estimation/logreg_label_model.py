
# 
# GNN_annot IJCNN 2021 implementation
#   Global label model implementation with an OvR logistic regression.
#   @author Viktor Varga
#

import numpy as np
from sklearn.linear_model import LogisticRegression

PARAM_MAX_ITER = 100  # default is 100
PARAM_TOL = 1e-4  # default is 1e-4
PARAM_REGULARIZER_WEIGHT = 2.  # default is 1.

class LogRegLabelModel():

    '''
    Member fields:
        n_cats: int; number of categories to classify into
        model: sklearn.linear_model.LogisticRegression
    '''

    def __init__(self):
        self.n_cats = None
        print("TODO, review logreg solver; default is changed from liblinear to lbfgs (original setup: liblinear with 'ovr' multi_class arg)")

    def __str__(self):
        return "LogReg"

    def reset(self, n_cats):
        '''
        Reinitialize model.
        '''
        self.n_cats = n_cats
        self.model = LogisticRegression(max_iter=PARAM_MAX_ITER, tol=PARAM_TOL, C=PARAM_REGULARIZER_WEIGHT, \
                                        solver='liblinear', multi_class='ovr')

    def fit(self, xs, ys):
        '''
        Parameters:
            xs: ndarray(n_sps, n_features) of float32; the feature vectors for each SP
            ys: ndarray(n_sps) of int32; the true labels of each SP
        '''
        assert xs.ndim == 2
        assert np.amax(ys) < self.n_cats
        self.model.fit(xs, ys)

    def predict(self, xs, return_probs=False):
        '''
        Parameters:
            xs: ndarray(n_sps, n_features) of float32; the feature vectors for each SP
            return_probs: bool;
        Returns:
            ys_pred: ndarray(n_sps, n_cat) of fl32 (IF return_probs == True)
                     ndarray(n_sps,) of i32        (IF return_probs == False)
        '''
        assert xs.ndim == 2
        if return_probs:
            ys_pred = self.model.predict_proba(xs)
        else:
            ys_pred = self.model.predict(xs)
        return ys_pred





