from typing import Dict

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes as mpl_Axes
from matplotlib import path
import matplotlib.pyplot as plt
#from scipy import ndimage
#import scipy.ndimage.filters


def plot_vertices(vertices: Dict[str, np.ndarray], show: bool = True) -> list[mpl_Axes]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    for i, key in enumerate(vertices):
        x = vertices[key][0]
        y = vertices[key][1]
        axes[i].plot(x, y, "-o", color="k", label=key)
        axes[i].set_title(key)
    if show:
        plt.show()
    return axes


# Laplacian operator (2nd order cetral-finite differences)
# dx, dy: sampling, w: 2D numpy array
def laplacian(dx, dy, w):
    """Calculate the laplacian of the array w=[]"""
    laplacian_xy = np.zeros(w.shape)
    for y in range(w.shape[1] - 1):
        laplacian_xy[:, y] = (1 / dy) ** 2 * (w[:, y + 1] - 2 * w[:, y] + w[:, y - 1])
    for x in range(w.shape[0] - 1):
        laplacian_xy[x, :] = laplacian_xy[x, :] + (1 / dx) ** 2 * (
            w[x + 1, :] - 2 * w[x, :] + w[x - 1, :]
        )
    return laplacian_xy


def create_grid(max_value: int = 5, n_steps=5):
    step = float(1 / n_steps)
    grid = np.mgrid[0:max_value:step, 0:max_value:step]
    return grid


def from_grid_to_points(grid):
    positions = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
    print(f"{positions=}")
    return positions


def interior_points(drum, grid):
    p1 = path.Path(drum.T)
    positions = from_grid_to_points(grid)
    mask = p1.contains_points(positions)
    return positions[mask].T


def main():

    drum1 = np.array([[0, 0, 2, 2, 3, 2, 1, 1, 0], [0, 1, 3, 2, 2, 1, 1, 0, 0]])
    drum2 = np.array([[1, 0, 0, 2, 2, 3, 2, 1, 1], [0, 1, 2, 2, 3, 2, 1, 1, 0]])

    vertices = {"drum1": drum1, "drum2": drum2}
    axes = plot_vertices(vertices, show=False)
    print("Cat")
    print(drum1)
    grid1 = create_grid(4)
    axes[0].scatter(grid1[0], grid1[1], color="gold")
    in_points1 = interior_points(drum1, grid1)
    axes[0].scatter(in_points1[0], in_points1[1], color="green")
    grid2 = create_grid(4)
    axes[1].scatter(grid2[0], grid2[1], color="red")
    in_points2 = interior_points(drum2, grid2)
    axes[1].scatter(in_points2[0], in_points2[1], color="green")
    plt.show()
    # scipy.ndimage.filters.laplace()


if __name__ == "__main__":
    main()
