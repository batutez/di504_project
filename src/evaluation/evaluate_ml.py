"""
Section 20 evaluate — evaluation for classical ML model

"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.evaluation.metrics import weighted_f1_with_ci, macro_auroc_with_ci


def evaluate_model_multiclass(model, X_test, y_test, class_names):
    """
    Calculate evaluation results one the provided ML classifier.

    Parameters
    ----------
    model : sklearn classifier
    X_test : feature test values
    y_test : target test values 
    class_names : in the same order as the label encoding

    Returns
    -------
    dict with keys: accuracy, weighted_f1, weighted_f1_ci, macro_auroc,
    macro_auroc_ci, confusion_matrix, classification_report, y_true,
    y_pred, y_proba.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1, weighted_f1_ci = weighted_f1_with_ci(y_test, y_pred)
    macro_auroc, macro_auroc_ci = macro_auroc_with_ci(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1, 'weighted_f1_ci': weighted_f1_ci,
        'macro_auroc': macro_auroc, 'macro_auroc_ci': macro_auroc_ci,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': np.asarray(y_test), 'y_pred': y_pred, 'y_proba': y_proba,
    }