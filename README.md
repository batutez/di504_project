# DI504 FINAL PROJECT

## Mental Health Text Classification

Replication and extension of Ding et al. (2025), "Trade-offs between machine learning and deep learning for mental illness detection on social media."

## Results

Multi-class classification of mental health status (7 classes) from social media text, using the Kaggle "Sentiment Analysis for Mental Health" dataset (52,681 statements). Five model configurations compared on an test set:


| Model                         | Weighted F1 |
| ----------------------------- | ----------- |
| ALBERT                        | 0.7777      |
| GRU + Adam + EDA augmentation | 0.7715      |
| GRU + AdamW                   | 0.7611      |
| GRU + Adam                    | 0.7583      |
| Logistic Regression           | 0.7375      |

Two improvements beyond the original paper: EDA based augmentation on the four smallest classes, and an AdamW vs. Adam comparison for the GRU. GRU hyperparameter search uses Optuna (TPE)

## Structure

```
src/                  preprocessing, models, training, evaluation code
data/                 raw + cached preprocessed data (gitignored)
outputs/              figures, results, model checkpoints (gitignored)

01_pipeline.ipynb        full pipeline: baseline, GRU, comparison, analysis
albert_finetuning.ipynb  ALBERT fine-tuning (run separately, GPU-heavy)
```

## How to run

1. Open "01_pipeline.ipynb" on Colab, mount Drive, run top to bottom (Sections 1–14, 18–20).
2. Open "albert_finetuning.ipynb" separately (Sections 15–17), loads the same cached split,  saves results back for Section 18.
