from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes as mpl_Axes
from matplotlib import path
import matplotlib.pyplot as plt

from dataclasses import dataclass

from scipy import ndimage

# import scipy.ndimage


@dataclass
class Polygon:
    path: path.Path

    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> Polygon:
        """Function that creates Polygon
        Args:
            vertices: 2D array of shape vertices. Last and first vert have to be same in order
             to have closed shape.
        """
        assert vertices.ndim == 2
        if vertices.shape[0] == 2:
            return cls(path.Path(vertices.T))
        elif vertices.shape[1] == 2:
            return cls(path.Path(vertices))
        else:
            raise RuntimeError(f"Vertices not in correct shape.\n{vertices.shape=}")

    @property
    def x_coords(self) -> np.ndarray:
        return self.path.vertices[:, 0]

    @property
    def y_coords(self) -> np.ndarray:
        return self.path.vertices[:, 1]


@dataclass
class Grid2D:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        assert self.x.ndim == 1
        assert self.y.ndim == 1
        assert self.x.shape[0] == self.y.shape[0]


def plot_shapes(shapes: Dict[str, Polygon], show: bool = True) -> list[mpl_Axes]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    for i, (shape_name, shape) in enumerate(shapes.items()):
        x = shape.x_coords
        y = shape.y_coords
        axes[i].plot(x, y, "-o", color="k", label=shape_name)
        axes[i].set_title(shape_name)
    if show:
        plt.show()
    return axes


# Laplacian operator (2nd order central-finite differences)
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
    return np.vstack([grid[0].ravel(), grid[1].ravel()]).T


def interior_points(grid: np.ndarray, polygon: Polygon) -> np.ndarray:
    positions = from_grid_to_points(grid)
    mask = polygon.path.contains_points(positions, radius=-1e-9)
    return positions[mask].T


def edge_points(grid: np.ndarray, polygon: Polygon) -> np.ndarray:
    positions = from_grid_to_points(grid)
    mask = polygon.path.contains_points(positions, radius=-1e-9)
    return positions[mask].T


def main():

    # Shape from the Matlab example.
    drum1_vertices = np.array([[0, 0, 2, 2, 3, 2, 1, 1, 0], [0, 1, 3, 2, 2, 1, 1, 0, 0]])
    # Rectangular shape.
    drum1_vertices = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    drum1_shape = Polygon.from_vertices(drum1_vertices)
    # drum2 = np.array([[1, 0, 0, 2, 2, 3, 2, 1, 1], [0, 1, 2, 2, 3, 2, 1, 1, 0]])

    # vertices = {"drum1": drum1, "drum2": drum2}
    axes = plot_shapes({"drum1": drum1_shape}, show=False)
    grid1 = create_grid(4, 5)
    axes[0].scatter(grid1[0], grid1[1], color="blue")
    in_points1 = interior_points(grid1, drum1_shape)
    axes[0].scatter(in_points1[0], in_points1[1], color="red")
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    xx, yy = np.meshgrid(in_points1[0], in_points1[1], sparse=False)
    print(f"{xx=}")
    print(f"{yy=}")
    laplace = ndimage.convolve(in_points1, stencil)
    print(f"{in_points1.shape=}")
    print(f"{in_points1[0].shape}")
    print(f"{laplace.shape=}")
    print(f"{laplace=}")
    # axes[1].scatter(in_points1[0], in_points1[1], color=laplace / laplace.max())
    axes[1].contour(in_points1[0], in_points1[1], laplace)
    # grid2 = create_grid(4)
    # axes[1].scatter(grid2[0], grid2[1], color="red")
    # in_points2 = interior_points(drum2, grid2)
    # axes[1].scatter(in_points2[0], in_points2[1], color="green")
    plt.show()


if __name__ == "__main__":
    main()
