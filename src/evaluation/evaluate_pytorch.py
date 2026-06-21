"""
Section 20 -  Evaluation for GRU and ALBERT
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.evaluation.metrics import weighted_f1_with_ci, macro_auroc_with_ci

def evaluate_pytorch_model(model, dataloader, device, class_names, use_attention_mask=False):
    """
    Run PyTorch model over a DataLoader and compute the full metrics suite.

    Returns
    -------
    dict — same shape as evaluate_model_multiclass's return value, plus
    y_proba = probs.
    """
    # before testing, we need to disable dropout and use running stats for batchnorm if any
    model.eval()  
    all_logits = []
    all_labels = []

    # no gradient tracking needed, this is done to save memory
    with torch.no_grad():  
        for batch in dataloader:
            if use_attention_mask:
                # ALBERT-style batch: dict with input_ids/attention_mask/label
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                # GRU-style batch  plain (sequences, labels) tuple
                sequences, labels = batch
                sequences, labels = sequences.to(device), labels.to(device)
                logits = model(sequences)

            # move off GPU immediately, batch by batch, rather than holding
            # the whole test set in GPU memory at once
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # concatanate every batchs logits and labels into single tensor for the test set
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # apply softmax to calculate probabilities and then use argmax to find y_pred
    probs = torch.softmax(all_logits, dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    y_true = all_labels.numpy()

    # calculate the metrics with y_true and y_pred
    accuracy = accuracy_score(y_true, y_pred)
    # weighted_f1_with_ci and macro_auroc_with_ci custom functions are in metrics.py
    weighted_f1, weighted_f1_ci = weighted_f1_with_ci(y_true, y_pred)
    macro_auroc, macro_auroc_ci = macro_auroc_with_ci(y_true, probs)
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1, 'weighted_f1_ci': weighted_f1_ci,
        'macro_auroc': macro_auroc, 'macro_auroc_ci': macro_auroc_ci,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true, 'y_pred': y_pred, 'y_proba': probs,
    }