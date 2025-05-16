from pathlib import Path
from typing import Optional, Tuple
import argparse
import sys

from matplotlib.collections import LineCollection
from tqdm import tqdm
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from colorama import Fore
import matplotlib.ticker as plticker
import matplotlib
from scipy.spatial.transform import Rotation

from .map import Map
from ...dataset import Dataset

# Note: This script has been modified to produce a specifiec figure for the paper.
# Revert to the previous commit to get nice plots for all locations.
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_map(map: Map) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(3.28125003459, 1.3))
    points = map.points

    # Rotate
    angle = -27.5
    r = Rotation.from_rotvec(
        np.array([0, 0, angle * np.pi / 180], dtype=np.float32)
    ).as_matrix()
    xyz = np.concatenate([points[:, -2:], np.zeros_like(points[:, :1])], axis=-1)
    points = (r @ xyz.T)[:2].T

    points -= points.min(axis=0)
    key_markers = points[np.where(map.is_keypoint == True)]
    intermediate_markers = points[np.where(map.is_keypoint == False)]

    # Plot the measurement errors.
    def get_index(marker: str) -> Optional[int]:
        try:
            return int(marker) - 1
        except:
            return None

    lines = []
    for measurement in map.distance_file.measurements:
        from_index = get_index(measurement.from_point)
        to_index = get_index(measurement.to_point)
        if (
            from_index is None
            or to_index is None
            or not isinstance(measurement.distance, float)
        ):
            continue
        from_point = key_markers[from_index]
        to_point = key_markers[to_index]
        optimized_distance = np.linalg.norm(from_point - to_point)
        lines.append(
            (from_point, to_point, abs(optimized_distance - measurement.distance))
        )

    discrepancies = np.array([line[-1] for line in lines])
    normalize = plt.Normalize(vmin=0, vmax=np.max(discrepancies))
    line_segments = LineCollection(
        [line[:2] for line in lines],
        linestyle="solid",
        colors=plt.cm.viridis(normalize(discrepancies)),
        linewidth=1,
    )
    ax.add_collection(line_segments)

    # Plot the points.
    ax.scatter(
        *key_markers.transpose(1, 0),
        s=40,
        zorder=100,
        color="#444",
        edgecolors="k",
        linewidths=1,
    )
    ax.scatter(
        *intermediate_markers.transpose(1, 0),
        s=20,
        zorder=200,
        color="#444",
        edgecolors="k",
        linewidths=1,
    )
    ax.set_aspect("equal")

    # Hide everything but the map itself.
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=2))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=2))
    ax.grid(color="#eee")
    ax.axis("equal")

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=normalize,
    )
    cbar = fig.colorbar(sm, ticks=np.arange(0, np.max(discrepancies), 0.002))
    cbar.ax.set_yticklabels([f"{int(x * 1000)} mm" for x in cbar.get_ticks()])

    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks_position("none")

    plt.subplots_adjust(
        top=1,
        bottom=0.04,
        left=0,
        right=0.985,
    )

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots of the dataset's locations."
    )
    parser.add_argument("root", type=Path, help="the dataset root path")
    parser.add_argument(
        "output", type=Path, help="the folder in which to save map plots"
    )
    args = parser.parse_args()
    output: Path = args.output

    dataset = Dataset(args.root)
    output.mkdir(parents=True, exist_ok=True)
    progress = tqdm(dataset.locations)
    for location in progress:
        if location.name != "SciLiShutters":
            print(f"Skipping {location.name}")
            continue

        if location.map is None:
            progress.write(
                f'{Fore.YELLOW}Warning: Location "{location.name}" does not have a map.{Fore.RESET}',
                file=sys.stderr,
            )
            continue
        fig, ax = plot_map(location.map)
        output_path = output / f"map_{location.name}.png"
        print(f"Saving to {output_path}")
        fig.savefig(output / f"map_{location.name}.pgf", dpi=300)
