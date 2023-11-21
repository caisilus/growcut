import numpy as np
import numba as nb
from numpy.linalg import norm
from math import sqrt

class GrowCut:
    def __init__(self, max_iterations=5):
        self.max_iterations = 5
        self.iteration = 0

        self.original_image = None

        self.previous_mask = None
        self.current_mask = None
        self.strengths = None

    def growcut(self, original, initial_mask):
        self.previous_mask = initial_mask
        self.current_mask = np.copy(initial_mask)
        self.original_image = original

        self.strengths = (np.any(initial_mask, axis=-1)).astype(np.float64)
        
        while (self.iteration < self.max_iterations):
            print("step")
            self.previous_mask = np.copy(self.current_mask)
            self.growcut_step()
            self.iteration += 1

        return self.current_mask

    def neighbours(self, x, y):
        neighbours = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if (x + dx >= self.shape()[0] or y + dy >= self.shape()[1] or (dx == 0 and dy == 0)):
                    continue
                neighbours.append((x+dx, y+dy))
        
        return neighbours

    # @nb.jit(nopython=True, parallel=True)
    def growcut_step(self):
        for x in range(self.original_image.shape[0]):
            for y in range(self.original_image.shape[1]):
                self.current_mask[x,y] = self.previous_mask[x,y]
                C = self.original_image[x, y]
                strength = self.strengths[x,y]

                for neighbour_x, neighbour_y in self.neighbours(x, y):
                    C_neighbour = self.original_image[neighbour_x, neighbour_y]
                    strength_neighbour = self.strengths[neighbour_x, neighbour_y]
                    new_strength = self.g(norm(C - C_neighbour)) * strength_neighbour
                    if new_strength > strength:
                        self.current_mask[x, y] = self.current_mask[neighbour_x, neighbour_y]
                        self.strengths[x, y] = new_strength
                
    def g(self, x):
        max_C = 255 * sqrt(3)
        return 1 - x/max_C

    def shape(self):
        return self.original_image.shape

    def C(self):
        return self.original_image