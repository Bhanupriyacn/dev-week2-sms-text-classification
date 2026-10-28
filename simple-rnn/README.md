# Simple RNN for SMS Text Classification

Predict **ham** (legitimate SMS) or **spam** from message text using the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Submission

- [Executed notebook on GitHub](https://github.com/Bhanupriyacn/dev-week2-sms-text-classification/blob/main/simple-rnn/Simple_RNN_SMS_Text_Classification_SUBMIT.ipynb)
- [Notebook on Kaggle](https://www.kaggle.com/code/bhanupriyacn/simple-rnn-sms-text-classification)
- Upload file: `Simple_RNN_SMS_Text_Classification_SUBMIT.ipynb`

## Exact assignment implementation

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

## Why long sequences are difficult

During backpropagation through time, gradients are repeatedly multiplied by recurrent weights and activation derivatives, which can make them shrink toward zero over many time steps (the vanishing gradient problem). As a result, words far back in a long sequence receive weak learning signals, making long-range dependencies difficult for a plain RNN to learn.

## Run the notebook

Install `requirements.txt`, open the notebook, and run all cells. On Kaggle, the source dataset is attached and internet is unnecessary. Locally or in Colab, place `spam.csv` beside the notebook or in `data/`; if absent, the loader downloads the source using KaggleHub. The dataset is downloaded from its source rather than redistributed in this repository. `Tokenizer` is a legacy Keras API retained to match the assignment.

- `Simple_RNN_SMS_Text_Classification_SUBMIT.ipynb`: complete executed submission.
- `requirements.txt`: runtime dependencies.
- `results_summary.json`: held-out local result and configuration.
- `training_history.csv`: all five local training epochs.

This RNN assignment is separate from the repository's earlier TF-IDF assignment.
