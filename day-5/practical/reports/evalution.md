## 3. Sentiment Classification Evaluation

The sentiment classification model is evaluated using Accuracy, Precision, Recall, and F1-score.

### Accuracy

Accuracy measures the percentage of correctly predicted sentiment labels.

`Accuracy = Correct Predictions / Total Predictions`

### Precision

Precision measures how many reviews predicted as a particular sentiment were actually correct.

`Precision = TP / (TP + FP)`

### Recall

Recall measures how many actual reviews of a particular sentiment were correctly identified.

`Recall = TP / (TP + FN)`

### F1-Score

F1-score is the harmonic mean of Precision and Recall.

`F1-Score = 2 × (Precision × Recall) / (Precision + Recall)`

### Classification Metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.50 (50%) |
| Precision (Weighted) | 0.50 (50%) |
| Recall (Weighted) | 0.50 (50%) |
| F1-Score (Weighted) | 0.50 (50%) |

The sentiment classification model achieved an accuracy of 50%. The weighted Precision, Recall, and F1-score were also 50%.

These results indicate that the model's predictions need improvement.

## 4. Semantic Search Evaluation Metrics

Semantic search is evaluated separately using:

- Precision@5
- Recall@5
- Average Rank of First Relevant Result
- Overlap@5

These metrics evaluate retrieval quality rather than sentiment classification accuracy.
