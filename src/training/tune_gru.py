"""
Sections 14-15 — GRU hyperparameter search (Optuna TPE) + Adam vs AdamW.

"""
import optuna
import random
import torch
import torch.nn as nn
from src.models.gru import GRUClassifier
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import copy

from src.evaluation.evaluate_pytorch import evaluate_pytorch_model
from config import (GRU_BATCH_SIZE, GRU_EMBEDDING_DIM_RANGE, GRU_HIDDEN_DIM_RANGE,
                     GRU_LR_RANGE, GRU_EPOCHS_RANGE,GLOBAL_SEED)

def _train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    Helper function used inside objective function.
    Performs forward pass, loss calculation, backpropagation,
    and optimizer updates for each batch. Returns the average
    training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sequences.size(0)
    return total_loss / len(loader.dataset)

def _evaluate_loss_and_f1(model, loader, criterion, device):
    """
    Helper function that calculate f1 score for validation dataset
    """
    # disable dropout 
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad(): # disable tracking for backprop
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            # calculate logits
            logits = model(sequences)
            # calcuulate validation loss
            loss = criterion(logits, labels)
            # accumulate total loss weighted by batch size
            total_loss += loss.item() * sequences.size(0)
            # save predictions
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            # save labels
            all_labels.extend(labels.cpu().tolist())
    # calculate average loss 
    avg_loss = total_loss / len(loader.dataset)
    # calculate f1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1

def make_objective(train_dataset, val_dataset, vocab_size, class_weights, device):
    """
    This function is a wrapper that creates optuna objective with provided 
    train, validation datasets, vocabulary size , class weights and device 
    All these parameters are used to create "objective" function.
    This function later build and return the optuna objective function
    Optuna calls this function once per trial

    Returns
    -------
    Callable[[optuna.Trial], float]
    """
    def objective(trial):
        """
        Optuna objective function for GRU hyperparameter optimization.
        Samples hyperparameters, trains a GRU model on the training set,
        evaluates it on the validation set, and returns the weighted F1 score
        to be maximized by Optuna.
        """
        # Hyperparameters (embedding dimension, hidden dimension, lr and epoch) are sampled from trial
        embedding_dim = trial.suggest_int('embedding_dim', *GRU_EMBEDDING_DIM_RANGE)
        hidden_dim = trial.suggest_int('hidden_dim', *GRU_HIDDEN_DIM_RANGE)
        lr = trial.suggest_float('lr', *GRU_LR_RANGE, log=True)
        epochs = trial.suggest_int('epochs', *GRU_EPOCHS_RANGE)

        # Building dataloaders 
        train_loader = DataLoader(train_dataset, batch_size=GRU_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=GRU_BATCH_SIZE, shuffle=False)
        # Creating the model , loss criterion and optimizer

        model = GRUClassifier(vocab_size, embedding_dim, hidden_dim).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loop over epochs

        for _ in range(epochs):
            # loop logic is hidden inside _train_one_epoch helper function
            # the functions mutate model that it takes its argument 
            _train_one_epoch(model, train_loader, optimizer, criterion, device)

        # after training is done, evaluation over validation is done and f1 score is calculated
        _, val_f1 = _evaluate_loss_and_f1(model, val_loader, criterion, device)
        return val_f1

    return objective


def run_gru_search(train_dataset, val_dataset, vocab_size, class_weights, device, n_trials=10):
    """
    Run the Optuna study with the TPE sampler.

    """
    random.seed(1234)
    sampler = optuna.samplers.TPESampler(seed=1234)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    objective = make_objective(train_dataset, val_dataset, vocab_size, class_weights, device)
    study.optimize(objective, n_trials=n_trials)

    print("Best params:", study.best_params)
    print("Best val weighted F1:", study.best_value)

    return study.best_params, study


def train_final_gru(best_params, train_dataset, val_dataset, vocab_size,
                     class_weights, device, optimizer_name="adam"):
    """
    Train final GRU model with best parameters coming from hyperparameter search
    This function can run with two different optimizers, adam and adamw
  
    """
   
    torch.manual_seed(GLOBAL_SEED)
    train_loader = DataLoader(train_dataset, batch_size=GRU_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=GRU_BATCH_SIZE, shuffle=False)

    model = GRUClassifier(vocab_size, best_params['embedding_dim'], best_params['hidden_dim']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name!r}. Use 'adam' or 'adamw'.")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(best_params['epochs']):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = _evaluate_loss_and_f1(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{best_params['epochs']} - train_loss: {train_loss:.4f}, "
              f"val_loss: {val_loss:.4f}, val_f1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)  # restore best-epoch weights before returning
    return model, train_losses, val_losses