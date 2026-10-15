# Day 6 NLP requirement check

Upload `Day_6_NLP_SMS_Text_Classification_SUBMIT.ipynb` from Downloads. The supplied dataset has ham/spam labels, so this notebook predicts those labels.

| Requirement | Notebook location | Verified evidence |
|---|---|---|
| RB1: Text preprocessing | Section 3 / code cell 8 | Lowercase, punctuation removal, tokenization, stopword removal, Porter stemming. |
| RB2: TF-IDF | Section 5 / code cell 12 | Explicit training fit_transform and held-out transform. |
| RB3: Supervised classifier | Section 5 / code cell 12 | LogisticRegression.fit on text features and true ham/spam labels. |
| RB4: Accuracy and F1 | Section 6 / code cell 14 | Accuracy, spam F1, macro F1 and weighted F1 on the test partition. |
| RB5: 10 predictive words per class | Section 7 / code cell 16 | Feature names mapped to signed fitted coefficients; exactly 10 words per class. |

Executed successfully: 19 cells, 3 plots, zero errors. Test accuracy: 98.38%; weighted F1: 0.9837; spam F1: 0.9339.
