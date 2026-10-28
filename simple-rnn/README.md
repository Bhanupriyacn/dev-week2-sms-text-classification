# Simple RNN for SMS Text Classification

Predict **ham** (legitimate SMS) or **spam** from message text using the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Corrected submission: Day-1 rubric evidence

The revised notebook addresses all five reported omissions and preserves the original SimpleRNN task. All 25 cells were executed locally without errors; four chart images are embedded in the notebook. This verification covers the supplied feedback; an external grade has not been rerun here.

| Rubric item | Explicit evidence in notebook |
|---|---|
| R1 — Initial inspection | Cell 4: `df.shape`, `head`, `dtypes`, `df.info()`, `df.describe(include="all")`, null/blank/duplicate counts; cell 5 interprets them. |
| R2 — Anomalies/outliers | Cell 10: character-length distribution, 1.5 × IQR bounds, flagged counts, longest-message inspection, missing/zero-word/conflicting-label checks; cell 11 explains retaining all 66 length outliers. |
| R3 — EDA visualizations | Cell 13: four rendered charts with titles and labelled axes: label counts, length histogram, length-by-label boxplot, spam proportion by length quartile. |
| R4 — Engineered dataframe features | Cells 7–8: `message_length`, `word_count`, `has_url`, `uppercase_ratio` in `eda_df`, with definitions, rationale, samples, dtypes and statistics by label. |
| R5 — Five insights | Cell 14: exactly five markdown bullets citing cleaning counts, class imbalance, length/word patterns, IQR outliers and URL association. |

The model's `df` still contains only `text` and `label`. EDA features live in `eda_df`; the RNN receives only 50-token text sequences. The model still uses `Embedding(5000, 32) → SimpleRNN(32) → Dense(1, sigmoid)` and runs exactly five epochs.

Files `eda_summary.json`, `features_by_label.csv`, `length_label_association.csv`, and `rubric_verification.json` provide inspectable supporting evidence. The four charts are embedded in the executed notebook and are also produced as PNG files when it runs.

## Submission

- [Executed notebook on GitHub](https://github.com/Bhanupriyacn/dev-week2-sms-text-classification/blob/main/simple-rnn/Simple_RNN_SMS_Text_Classification_SUBMIT.ipynb)
- [Notebook on Kaggle](https://www.kaggle.com/code/bhanupriyacn/simple-rnn-sms-text-classification)
- Upload file: `Simple_RNN_SMS_Text_Classification_SUBMIT.ipynb`

## Exact RNN assignment implementation

Keep only `text` and `label`, encode ham=0 and spam=1, and fit Keras `Tokenizer(num_words=5000, oov_token="<OOV>")` on training messages only. Use `pad_sequences(..., maxlen=50)` for both partitions, with pre-padding and pre-truncation. Every model input has exactly 50 token indices, irrespective of original message length.

```python
model = Sequential([
    Embedding(5000, 32),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, epochs=5, verbose=2)
```

The model has 162,113 parameters. The dataset has two classes, so the output is sigmoid; for more than two classes, use `Dense(n_classes, activation="softmax")` with sparse categorical cross-entropy and integer labels.

## Data separation and results

Remove 403 exact duplicate messages before the stratified 80/20 split (`random_state=42`); retain 5,169 unique messages. The tokenizer never fits on test text. All five epochs run without early stopping or tuning against the test set.

| Measure | Local executed result |
|---|---:|
| Training messages | 4,135 |
| Test messages | 1,034 |
| Epochs | 5 |
| Test accuracy | 97.68% |
| Majority-class baseline | 87.33% |
| Spam precision | 1.0000 |
| Spam recall | 0.8168 |
| Spam F1 | 0.8992 |

The confusion matrix (rows=actual, columns=predicted; order=ham, spam) is `[[903, 0], [24, 107]]`. The notebook includes all training epochs, model summary, test accuracy, and the full classification report. Accuracy alone can hide missed spam; 24 of 131 test spam messages were missed in this local run. Results describe one historical dataset split and may vary with TensorFlow version and hardware; local execution used TensorFlow 2.20.0.

Kaggle Version 1 also completed successfully with the same split and required settings: **97.49% test accuracy**, confusion matrix `[[903, 0], [26, 105]]`. Small numerical differences across CPU/platform runs can change training even with fixed seeds; the local executed notebook reports **97.68%**. See `kaggle_results_summary.json` for the independently executed Kaggle result.

## Why long sequences are difficult

During backpropagation through time, gradients are repeatedly multiplied by recurrent weights and activation derivatives, which can make them shrink toward zero over many time steps (the vanishing gradient problem). As a result, words far back in a long sequence receive weak learning signals, making long-range dependencies difficult for a plain RNN to learn.

## Run the notebook

Install `requirements.txt`, open the notebook, and run all cells. On Kaggle, the source dataset is attached and internet is unnecessary. Locally or in Colab, place `spam.csv` beside the notebook or in `data/`; if absent, the loader downloads the source using KaggleHub. The dataset is downloaded from its source rather than redistributed in this repository. `Tokenizer` is a legacy Keras API retained to match the assignment.

- `Simple_RNN_SMS_Text_Classification_SUBMIT.ipynb`: complete executed submission.
- `requirements.txt`: runtime dependencies.
- `results_summary.json`: held-out local result and configuration.
- `training_history.csv`: all five local training epochs.

This RNN assignment is separate from the repository's earlier TF-IDF assignment.
