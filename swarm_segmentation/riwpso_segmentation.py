import numpy as np
import pyswarms as ps
import cv2

def fitness_function(particles, image):
    costs = []
    for p in particles:
        thresh = int(abs(p[0]) % 255)
        _, segmented = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        cost = np.var(segmented)
        costs.append(-cost)  # maximize variance
    return np.array(costs)

def run_riwpso(image_path, n_particles=20, iters=40):
    image = cv2.imread(image_path, 0)

    def objective(x):
        return fitness_function(x, image)

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=1,
        options=options
    )

    best_cost, best_pos = optimizer.optimize(objective, iters=iters)
    threshold = int(best_pos[0]) % 255
    _, segmented = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return segmented
