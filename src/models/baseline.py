"""
Sections 10-11 — Baseline models, Logistic Regression.


"""
from sklearn.linear_model import LogisticRegression
from config import GLOBAL_SEED

LR_PARAM_GRID = {
    "C" : [0.1,1,10],
    "solver" : ["lbfgs", "saga"],
    "penalty" : ["l2"],
    "class_weight" : ["balanced" , None],
    "max_iter" : [1000]
}


def build_logistic_regression(params):
    """
    Construct a LogisticRegression instance from one hyperparameter
    combination (a dict with keys matching LR_PARAM_GRID).
    """
    model = LogisticRegression(**params, random_state = GLOBAL_SEED)
    return model
