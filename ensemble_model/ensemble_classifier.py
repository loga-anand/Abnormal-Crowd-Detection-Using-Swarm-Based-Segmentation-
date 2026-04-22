from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_ensemble(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    svm = SVC(kernel="rbf", probability=True)
    rf = RandomForestClassifier(n_estimators=100)

    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    print("[INFO] Ensemble Classification Report:")
    print(classification_report(y_test, y_pred))

    # 🔥 SAVE MODEL HERE
    os.makedirs("models", exist_ok=True)
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    print("[INFO] Trained model saved to models/ensemble_model.pkl")

    return ensemble, y_test, y_pred, y_prob
