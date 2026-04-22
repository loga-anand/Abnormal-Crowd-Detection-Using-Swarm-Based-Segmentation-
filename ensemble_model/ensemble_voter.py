import numpy as np

class EnsembleVoter:
    def __init__(self, weights=None):
        self.weights = weights or {
            "rule": 0.3,
            "svm": 0.25,
            "rf": 0.25,
            "lstm": 0.2
        }

    def vote(self, rule_score, svm_score, rf_score, lstm_score):
        final_score = (
            self.weights["rule"] * rule_score +
            self.weights["svm"] * svm_score +
            self.weights["rf"] * rf_score +
            self.weights["lstm"] * lstm_score
        )
        return int(final_score >= 0.5)
