# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 11:19:02 2025

@author: craigm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point
import geopandas as gpd
from pathlib import Path

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)


def tukey_depth_2d(point, points):
    points = np.asarray(points)
    point = np.asarray(point)

    if len(points) == 0:
        return 0

    rel = points - point
    rel = rel[~np.all(rel == 0, axis=1)]

    if len(rel) == 0:
        return 0

    angles = np.arctan2(rel[:, 1], rel[:, 0])
    angles = np.sort(angles)
    angles = np.concatenate([angles, angles + 2 * np.pi])

    n = len(rel)
    min_count = n
    j = 0

    for i in range(n):
        while angles[j] < angles[i] + np.pi:
            j += 1
        min_count = min(min_count, j - i)

    return min_count


coastline = (
    Path(__file__).resolve().parent
    / "data"
    / "Coastlines_and_Islands_Polygons_50K.shp"
)

coast = gpd.read_file(coastline)

data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"

vents = pd.read_csv(data_file, comment="#")

vents = vents.dropna(subset=["easting", "northing"]).reset_index(drop=True)

coords = vents[["easting", "northing"]].values

depths = []

for i in range(len(coords)):
    d = tukey_depth_2d(coords[i], coords)
    depths.append(d)

vents["TukeyDepth"] = depths

# Tukey median = point(s) with maximum depth
max_depth = vents["TukeyDepth"].max()
median_points = vents[vents["TukeyDepth"] == max_depth]

# Bag definition: points with depth >= 50% of max
depth_threshold = 0.5 * max_depth
bag_points = vents[vents["TukeyDepth"] >= depth_threshold]

# Compute convex hull of bag
bag_hull = MultiPoint(bag_points[["easting", "northing"]].values).convex_hull

# Choose a single median point (centroid if multiple)
median_xy = median_points[["easting", "northing"]].mean().values

inflation_factor = 3.0

fence_points = []
for x, y in bag_points[["easting", "northing"]].values:
    vec = np.array([x, y]) - median_xy
    fence_points.append(median_xy + inflation_factor * vec)

fence_hull = MultiPoint(fence_points).convex_hull


def outside_fence(row, fence):
    p = Point(row["easting"], row["northing"])
    return not (fence.contains(p) or fence.touches(p))


vents["Outlier"] = vents.apply(outside_fence, axis=1, fence=fence_hull)

fig, ax = plt.subplots(figsize=(8, 8))

# All vents
ax.scatter(
    vents["easting"], vents["northing"], c="k", marker="^", s=30, label="Vents"
)

# Outliers
outliers = vents[vents["Outlier"]]
ax.scatter(
    outliers["easting"],
    outliers["northing"],
    c="red",
    marker="^",
    s=50,
    label="Outside fence",
)

# Bag hull
if bag_hull.geom_type == "Polygon":
    xb, yb = bag_hull.exterior.xy
    ax.fill(xb, yb, color="lightblue", alpha=0.5, label="50% bag")
    ax.plot(xb, yb, color="blue", lw=1)

# Fence hull
if fence_hull.geom_type == "Polygon":
    xf, yf = fence_hull.exterior.xy
    ax.plot(xf, yf, "k--", lw=1, label="Fence")

center_point = (1761500, 5914950)
ax.scatter(
    center_point[0], center_point[1], c="k", s=90, label="ellipse center"
)

# Median
ax.scatter(
    median_xy[0],
    median_xy[1] - 100,
    edgecolors="k",
    s=120,
    marker="o",
    facecolors="none",
    label="Tukey median",
)
coast.plot(ax=ax, linewidth=1, zorder=0, color="gray", alpha=0.2)

ax.set_aspect("equal")
ax.legend()
ax.set_xlim(1745000, 1780000)
ax.set_ylim(5896500, 5933500)
ax.set_ylabel("Northing NZTM [m]")
ax.set_xlabel("Easting NZTM [m]")
plt.savefig(
    data_file=Path(__file__).resolve().parent / "Figure5_AVF_bagplot.png",
    dpi=300,
)
