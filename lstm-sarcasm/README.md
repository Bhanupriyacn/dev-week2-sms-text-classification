# LSTM for Text Classification — Sarcasm Headlines

## Problem statement

Use the [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) to classify headlines as **not sarcastic (0)** or **sarcastic (1)**. Reuse Day 10's Keras tokenizer with the top-5000 vocabulary limit and `pad_sequences(..., maxlen=50)`. Build exactly `Embedding(5000, 32) → LSTM(32) → Dense(1, activation="sigmoid")`, train for exactly **5 epochs**, and report test accuracy. Explain the forget gate in one or two sentences. A separate notebook performs the optional LSTM-to-GRU swap.

## Submission files and links

- [Required executed LSTM notebook on GitHub](https://github.com/Bhanupriyacn/dev-week2-sms-text-classification/blob/main/lstm-sarcasm/LSTM_Sarcasm_Text_Classification_SUBMIT.ipynb)
- [Required LSTM notebook on Kaggle](https://www.kaggle.com/code/bhanupriyacn/lstm-sarcasm-headlines-classification)
- [Bonus executed GRU notebook on GitHub](https://github.com/Bhanupriyacn/dev-week2-sms-text-classification/blob/main/lstm-sarcasm/GRU_Sarcasm_BONUS.ipynb)
- [Bonus GRU notebook on Kaggle](https://www.kaggle.com/code/bhanupriyacn/gru-sarcasm-headlines-bonus)

**Required upload:** `LSTM_Sarcasm_Text_Classification_SUBMIT.ipynb`. The GRU notebook is optional and separate.

## Dataset and preprocessing

Use **Sarcasm_Headlines_Dataset_v2.json alone**, not the overlapping original file. The source has 28,619 rows and three columns. Keep only `headline` and `is_sarcastic`, renamed `text` and `label`; exclude `article_link` from input. Inspection shows no missing or blank text/label values and no conflicting labels. Remove 116 exact duplicate headlines, leaving 28,503 unique examples.

A stratified 80/20 split (`random_state=42`) gives **22,802 train / 5,701 test** headlines. Fit `Tokenizer(num_words=5000, oov_token="<OOV>")` only on training text. Use its default lowercasing/punctuation filtering and pre-pad/pre-truncate both partitions to exactly **50 tokens**. Feature engineering is confined to `eda_df`; the neural model receives text token sequences only.

## Required exact model

```python
model = Sequential([
    Embedding(5000, 32),
    LSTM(32),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, epochs=5, verbose=2)
```

This changes only Day 10's recurrent layer. Batch size remains 64, the seed remains 42, and there are no extra layers, dropout, early stopping, or test-set tuning. The LSTM model has **168,353 parameters**. For more than two classes use a `Dense(n_classes, activation="softmax")` output with sparse categorical cross-entropy and integer labels.

## Local held-out results

| Model | Test accuracy | Test loss | Epochs | Parameters |
|---|---:|---:|---:|---:|
| Required LSTM | 85.51% | 0.3638 | 5 | 168,353 |
| Bonus GRU | 85.72% | 0.3705 | 5 | 166,369 |
| Majority-class baseline | 52.45% | — | — | — |

GRU differs by **0.21 percentage points** in these local runs. Its copy uses the same data, tokenizer, split, padding, seed, optimizer, batch size, and epoch count; only the recurrent layer and related parameter assertion/output folder change. One run per model does not establish general superiority. Results on this dataset cannot be directly compared with earlier SMS accuracy to isolate the effect of LSTM.

LSTM confusion matrix (rows=actual, columns=predicted; order=not sarcastic, sarcastic): `[[2632, 358], [468, 2243]]`. The notebooks also report per-class precision, recall and F1. These results describe a fixed holdout from this historical source dataset; floating-point/platform differences can slightly change training outcomes. Both local runs use TensorFlow 2.20.0.

## EDA rubric evidence included in both notebooks

| Item | Explicit notebook evidence |
|---|---|
| R1 | Cells 4–5: shape, sample, dtypes, info, describe-all, missing/blank/duplicate checks and interpretation. |
| R2 | Cells 10–11: length-IQR bounds 10–114 characters, 9 short and 295 long outliers, shortest/longest examples, and justified retention without feature capping. |
| R3 | Cell 13: four rendered labelled charts—label counts, length histogram, length-by-label boxplot, and sarcasm proportion by length quartile. |
| R4 | Cells 7–8: `message_length`, `word_count`, `punctuation_count`, `has_question`, with definitions, rationale, examples, dtypes and feature statistics by label. |
| R5 | Cell 14: five markdown bullets with observed cleaning counts, class balance, length/verbosity, anomalies and question-mark association. |

The 304 length outliers are retained because length alone does not invalidate source headlines. The model separately enforces the assignment's fixed 50-token input limit. Each executed notebook contains **25 cells**, **4 embedded chart images**, and **0 execution errors**. This verifies the supplied rubric evidence; the external grader has not been run.

## Forget-gate explanation

An LSTM's forget gate learns how much of each value in its previous cell state to retain or erase, so it can discard irrelevant information while preserving useful context over many time steps. A plain RNN has no separate gated cell-state path for this selective retention, making long-term information harder to preserve.

## Reproduction and attribution

Install `requirements.txt` and run all cells. On Kaggle, attach the linked dataset; internet is not required for execution. Locally/Colab, put the v2 JSON beside the notebook or in `data/`; otherwise the loader downloads it with KaggleHub. `Tokenizer` is a legacy Keras API retained to match the assignment. Rishabh Misra's source dataset is CC BY 4.0 according to Kaggle metadata; it is attached/downloaded from the source instead of redistributed in this repository.

Supporting files include local model summaries, all five epochs of training history, the EDA summary, feature/association tables, and the rubric-verification report. The source and split assertions ensure that the five insight bullets match the loaded v2 data.
