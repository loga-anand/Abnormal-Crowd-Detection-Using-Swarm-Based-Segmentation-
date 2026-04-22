import numpy as np

class FeatureFusion:
    @staticmethod
    def fuse(cnn_features, motion_features):
        fused = []

        for i in range(min(len(cnn_features), len(motion_features))):
            motion_vals = list(motion_features[i].values())
            combined = np.concatenate([
                cnn_features[i].flatten(),
                np.array(motion_vals, dtype=np.float32)
            ])
            fused.append(combined)

        return np.array(fused)
