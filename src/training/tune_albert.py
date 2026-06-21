"""
Section 18 — ALBERT random search for hyperparameter tuning
"""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification, AlbertConfig
from sklearn.metrics import f1_score
from config import ALBERT_BATCH_SIZE, ALBERT_LR_RANGE, ALBERT_EPOCHS_RANGE, ALBERT_DROPOUT_RANGE
import pickle
from config import MODELS_DIR, RESULTS_DIR,ALBERT_MODEL_NAME,NUM_CLASSES

def sample_albert_hyperparams():
    """
    Draw one random hyperparameter combination from the ALBERT search space.

    Samples:
        lr: log-uniform between 1e-5 and 1e-4
        epochs: random integer in [3, 5]
        dropout: uniform between 0.1 and 0.5
          
    Returns
    -------
    dict with keys: 'lr', 'epochs', 'dropout'
    """
    log_lr = np.random.uniform(np.log10(ALBERT_LR_RANGE[0]), np.log10(ALBERT_LR_RANGE[1]))
    lr = 10 ** log_lr
    epochs = np.random.randint(ALBERT_EPOCHS_RANGE[0], ALBERT_EPOCHS_RANGE[1] + 1)
    dropout = np.random.uniform(ALBERT_DROPOUT_RANGE[0], ALBERT_DROPOUT_RANGE[1])
    return {'lr': lr, 'epochs': epochs, 'dropout': dropout}


def train_and_evaluate_albert(params, train_dataset, val_dataset, class_weights, device):
    """
    Build one ALBERT model with params['dropout'] as hidden_dropout_prob,
    fine-tune it for params['epochs'] epochs at params['lr'], and return its
    weighted F1 on val_dataset.

    Returns
    -------
    (model, val_weighted_f1, train_losses: list[float], val_losses: list[float])
        val_weighted_f1 is the LAST epoch's val F1, since this function
        trains for exactly params['epochs'] epochs and doesn't keep a
        separate best-epoch checkpoint within a single trial.
    """
    # create config for custom Albert model
    config = AlbertConfig.from_pretrained(ALBERT_MODEL_NAME, num_labels=NUM_CLASSES,
                                           hidden_dropout_prob=params['dropout'])
    
    # create the model send to device
    model = AlbertForSequenceClassification.from_pretrained(ALBERT_MODEL_NAME, config=config).to(device)

    # create dataloader from training and validation set
    train_loader = DataLoader(train_dataset, batch_size=ALBERT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ALBERT_BATCH_SIZE, shuffle=False)

    # loss function is Cross Entropy Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

    train_losses, val_losses = [], []

    for epoch in range(params['epochs']):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * input_ids.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_val_loss += loss.item() * input_ids.size(0)
                all_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch+1}/{params['epochs']} - train_loss: {avg_train_loss:.4f}, "
              f"val_loss: {avg_val_loss:.4f}, val_f1: {val_f1:.4f}")

    return model, val_f1, train_losses, val_losses


def run_albert_search(train_dataset, val_dataset, class_weights, device, n_iter=10):
    """
    Run the full random search loop and keep the best model.

    Returns
    -------
    (best_model, best_params, best_f1, train_losses, val_losses, results)
    results : list[tuple[dict, float]]
        Every trial's (params, val_f1), needed for the search-visualization
        plots in the notebook (no Optuna study object exists for plain
        random search).
    """



    random.seed(1234)
    np.random.seed(1234)

    best_f1 = -1
    best_model = None
    best_params = None
    best_train_losses = None
    best_val_losses = None
    results = []

    for i in range(n_iter):
        params = sample_albert_hyperparams()
        print(f"\n--- Trial {i+1}/{n_iter}: {params} ---")
        try:
            model, val_f1, train_losses, val_losses = train_and_evaluate_albert(
                params, train_dataset, val_dataset, class_weights, device
            )
        except Exception as e:
            print(f"Trial {i+1} failed: {e}")
            results.append((params, None))
            continue

        results.append((params, val_f1))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_params = params
            best_train_losses = train_losses
            best_val_losses = val_losses
            # Save the best-so-far model immediately, not just at the end —
            # protects against losing everything if Colab disconnects later.
            torch.save(best_model.state_dict(), MODELS_DIR / "best_albert_model.pth")

        # Save results-so-far after every trial, regardless of whether it improved —
        # this is what lets you recover and see partial progress if the session dies.
        with open(RESULTS_DIR / "albert_search_results.pkl", "wb") as f:
            pickle.dump(results, f)

    print("\nAll trials, sorted by val F1:")
    valid_results = [(p, f1) for p, f1 in results if f1 is not None]
    for params, f1 in sorted(valid_results, key=lambda x: x[1], reverse=True):
        print(f"{params} -> val F1: {f1:.4f}")

    print(f"\nBest model saved to {MODELS_DIR / 'best_albert_model.pth'}")

    return best_model, best_params, best_f1, best_train_losses, best_val_losses, results