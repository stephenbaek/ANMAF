# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:32:39 2019

@author: sbaek
"""
import numpy as np

def euclidean_distance(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=np.random.uniform):
    tau = 2 * np.pi
    cellsize = r / np.sqrt(2)

    grid_width = int(np.ceil(width / cellsize))
    grid_height = int(np.ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(np.floor(p[0] / cellsize)), int(np.floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * np.sqrt(3 * random() + 1)
            px = qx + d * np.cos(alpha)
            py = qy + d * np.sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    return np.array([p for p in grid if p is not None]).astype('int32')
