import numpy as np
import pyswarms as ps

def fitness_function(particles):
    return np.sum(particles ** 2, axis=1)

def run_pso():
    optimizer = ps.single.GlobalBestPSO(
        n_particles=30,
        dimensions=2,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    )

    best_cost, best_pos = optimizer.optimize(fitness_function, iters=50)
    return best_pos
