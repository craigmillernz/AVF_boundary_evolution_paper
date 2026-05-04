# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 09:07:39 2025

@author: craigm
"""

from shapely.geometry import Point, MultiPoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as tck
from pathlib import Path

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"
vents = pd.read_csv(data_file, comment="#")

vents = vents.sort_values(by="Hopkins2020_Age")


# Ensure sorted by age (oldest → youngest)
vents = vents.sort_values("Hopkins2020_Age", ascending=False).reset_index(
    drop=True
)
vents.dropna(subset="Hopkins2020_Age", inplace=True)


# Probability all data
areas = []
ages = []
pts = []

# --- Iteratively build convex hull and compute area ---
for i, row in vents.iterrows():
    p = Point(row["easting"], row["northing"])
    pts.append(p)

    # Only compute hull once we have >= 3 points
    if len(pts) >= 3:
        hull = MultiPoint(pts).convex_hull
        area = hull.area / 1e6  # convert m² → km²
    else:
        area = 0.0

    areas.append(area)
    ages.append(row["Hopkins2020_Age"])

# Convert to DataFrame if needed
hull_evolution = pd.DataFrame(
    {"Age_ka": ages, "HullArea_km2": areas, "Name": vents.name}
)

# Sort by age (optional, depending on plotting preference)
hull_evolution = hull_evolution.sort_values("Age_ka", ascending=False)


# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax2 = ax.twinx()

plot1 = ax.plot(
    hull_evolution["Age_ka"],
    hull_evolution["HullArea_km2"],
    color="green",
    linewidth=1.2,
    label="AVF Area",
)

ax.scatter(
    hull_evolution["Age_ka"],
    hull_evolution["HullArea_km2"],
    color="green",
    s=18,
    edgecolors="k",
    linewidth=0.4,
    zorder=3,
)

ax.set_xlabel("Eruption Age [ka]")
ax.set_ylabel("Convex Hull Area [km²]")
ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
# for i, txt in enumerate(hull_evolution.Name):
#    ax.text(x=hull_evolution["Age_ka"][i], y=hull_evolution["HullArea_km2"][i], s=hull_evolution["Name"][i], rotation=90)

rect1 = Rectangle(
    (60, 215),
    width=140,
    height=-220,
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1,
)
ax.add_patch(rect1)

rect2 = Rectangle(
    (15, 207),
    width=45,
    height=115.0,
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1,
)

ax.add_patch(rect2)

rect3 = Rectangle(
    (0, 350),
    width=15,
    height=-40,
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1,
)

ax.add_patch(rect3)
ax.invert_xaxis()
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=200, xmax=0)


def tukey_depth_2d(point, points):
    """
    Compute exact Tukey (halfspace) depth of a 2D point
    relative to a set of 2D points using angular sweep.

    Parameters
    ----------
    point : (2,) array-like
        Point at which depth is evaluated
    points : (N,2) array-like
        Reference point cloud

    Returns
    -------
    depth : int
        Tukey depth (integer count)
    """
    points = np.asarray(points)
    point = np.asarray(point)

    if len(points) == 0:
        return 0

    # Shift points relative to target point
    rel = points - point

    # Remove identical points (zero vectors)
    rel = rel[~np.all(rel == 0, axis=1)]

    if len(rel) == 0:
        return 0

    # Compute angles
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    angles = np.sort(angles)

    # Duplicate angles shifted by 2π for circular sweep
    angles = np.concatenate([angles, angles + 2 * np.pi])

    # Sliding window over half-circle
    n = len(rel)
    min_count = n
    j = 0

    for i in range(n):
        while angles[j] < angles[i] + np.pi:
            j += 1
        count = j - i
        min_count = min(min_count, count)

    return min_count


records = []

for i in range(len(vents)):
    # Older vents up to this eruption
    subset = vents.iloc[: i + 1]

    coords = subset[["easting", "northing"]].values
    newest = coords[-1]

    depth = tukey_depth_2d(newest, coords[:-1])

    records.append(
        {
            "Age_ka": vents.loc[i, "Hopkins2020_Age"],
            "TukeyDepth": depth,
            "N_vents": i + 1,
        }
    )

depth_time = pd.DataFrame(records)

plot2 = ax2.plot(
    depth_time["Age_ka"],
    depth_time["TukeyDepth"],
    color="steelblue",
    linewidth=1.2,
    label="Tukey depth",
)

ax2.scatter(
    depth_time["Age_ka"],
    depth_time["TukeyDepth"],
    color="steelblue",
    s=18,
    edgecolors="k",
    linewidths=0.4,
    zorder=3,
)

ax2.set_ylabel("Tukey Depth")
ax2.set_yticks(range(0, 20, 2))
ax2.minorticks_on()

ax2.yaxis.set_minor_locator(tck.AutoMinorLocator(2))

lines = plot1 + plot2
labels = [l.get_label() for l in lines]

ax.legend(lines, labels, loc="upper left")
ax2.set_ylim(ymin=0)

ax.text(
    vents["Hopkins2020_Age"].iloc[0],
    10,
    "Pupuke Moana",
    rotation=90,
    horizontalalignment="center",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[3],
    30,
    "Whakamuhu",
    rotation=90,
    horizontalalignment="center",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[4],
    50,
    "Albert Park",
    rotation=90,
    horizontalalignment="center",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[5] - 2,
    85,
    "Pukewairiki",
    verticalalignment="center",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[5] + 5,
    120,
    "Boggust Park",
    rotation=90,
    verticalalignment="bottom",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[7],
    130,
    "Ōrākei Basin",
    rotation=90,
    horizontalalignment="center",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[18],
    248,
    "Hampton Park",
    rotation=90,
    horizontalalignment="right",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[19],
    250,
    "Te Puke o Taramainuku",
    rotation=90,
    horizontalalignment="left",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[20],
    285,
    r"Matukutūreia",
    rotation=90,
    horizontalalignment="center",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[39],
    325,
    "Motukorea",
    rotation=90,
    horizontalalignment="center",
)
ax.text(
    vents["Hopkins2020_Age"].iloc[50] + 1,
    295,
    "Rangitoto",
    rotation=90,
    horizontalalignment="center",
)

plt.savefig(
    data_file=Path(__file__).resolve().parent
    / "Figure4_avf_area_vs_age_tukey_depth_combined.png",
    dpi=300,
)
