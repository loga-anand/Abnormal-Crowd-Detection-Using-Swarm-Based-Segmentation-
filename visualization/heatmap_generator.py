import cv2
import numpy as np

def generate_heatmap(frame, motion_map):
    heat = cv2.normalize(motion_map, None, 0, 255, cv2.NORM_MINMAX)
    heat = heat.astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
