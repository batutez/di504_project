"""
Sections 10-11 — Manual grid search for Logistic Regression and SVM.

"""
import itertools
import random
from sklearn.metrics import f1_score
from src.models.baseline import build_logistic_regression, build_svm, LR_PARAM_GRID, SVM_PARAM_GRID
from config import HYPERPARAM_SEARCH_SEED

def grid_search(build_fn, param_grid, X_train, y_train, X_val, y_val):
   """
   Generic manual grid search, which tries every combination in param_grid, fit on
   train, score on val with weighted F1, keep the best.

   Uses itertools.product instead of GridSearchCV. It is how it is done in the reimplemented paper
   and this version gives full control to evaluate on a fix validation set 

   Returns
   -------
   (best_model, best_params: dict, best_val_f1: float)

   """
   random.seed(HYPERPARAM_SEARCH_SEED)
   # exract the parameter names
   keys = list(param_grid.keys())
   # create the parameter combinations with itertools.product
   combinations = list(itertools.product(*param_grid.values()))
   print(f"Trying {len(combinations)} combinations...")
   # setting max_score to -1, this will be updated inside the for loop
   max_score = -1

   best_model = None
   best_params = None

   for combination in combinations:
      # for each combination of parameters create the param_dict
      param_dict = {k:v for k,v in zip(keys,combination)}
      try: 
         # build the model with param combinaion
         model = build_fn(param_dict)
         # fit, predict values and calculate f1 score with validation data
         model.fit(X_train,y_train)
         preds = model.predict(X_val)
         score = f1_score(y_val, preds, average='weighted')
      except Exception as e:
         print(f"Skipped {param_dict}: {e}")
         continue
      print(f"{param_dict} -> weighted F1: {score:.4f}")

      # update logic, if score is bigger than the existing max_score update best model and parameters
      if score > max_score:
         max_score = score
         best_model = model
         best_params = param_dict
      
   return best_model, best_params, max_score