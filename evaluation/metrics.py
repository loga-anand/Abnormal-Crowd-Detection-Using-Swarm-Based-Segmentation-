from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob=None):
    print("\n[INFO] Evaluation Metrics")

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))

    if y_prob is not None:
        print("ROC-AUC  :", roc_auc_score(y_true, y_prob))

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
