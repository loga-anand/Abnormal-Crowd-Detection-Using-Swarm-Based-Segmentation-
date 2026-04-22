from ensemble_model.build_dataset import build_feature_dataset
from ensemble_model.ensemble_classifier import train_ensemble

X, y = build_feature_dataset()

print("[INFO] Feature matrix shape:", X.shape)
print("[INFO] Labels shape:", y.shape)

model = train_ensemble(X, y)

print("[INFO] Ensemble training completed successfully")
