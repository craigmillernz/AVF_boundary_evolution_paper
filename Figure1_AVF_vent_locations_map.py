# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 08:39:06 2025

@author: craigm
"""

import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from scipy.spatial import ConvexHull

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = "D:/Dropbox/AVF/data/AVF_main_vents.csv"
coastline = "D:/Dropbox/GIS/NZTM/Coastlines_and_Islands_Polygons_50K.shp"

coast = gpd.read_file(coastline)

vents = pd.read_csv(data_file, comment="#", encoding="utf-8")
vents_all = pd.read_csv(data_file, comment="#", encoding="utf-8")

# normalize to precomposed macron characters (some entries use combining diacritics)
for df in (vents, vents_all):
    for col in ("name", "Reo"):
        df[col] = df[col].apply(
            lambda v: unicodedata.normalize("NFC", v) if isinstance(v, str) else v
        )

vents = vents.sort_values(by="Hopkins2020_Age")

# Ensure sorted by age (oldest → youngest)
vents = vents.sort_values("Hopkins2020_Age", ascending=True).reset_index(drop=True)
vents.dropna(subset="Hopkins2020_Age", inplace=True)
vents["age_rank"] = vents["Hopkins2020_Age"].rank(method="min", ascending=False)

volume = vents.dropna(subset="volume")
names = vents.sort_values("Hopkins2020_Age")
names.drop(columns=["type", "Age_note", "easting", "northing", "description"])

no_age = vents_all[vents_all["Hopkins2020_Age"].isna()]
# %%
# plot
center_point = (1761500, 5914950)  # (1761475, 5914911)
width = 17000  # 16481
height = 29300  # 28884
angle = 0  # 2.6


fig, ax = plt.subplots(figsize=(11, 10))
coast.plot(ax=ax, linewidth=1, zorder=0, color="gray", alpha=0.2)

ellipse = patches.Ellipse(
    xy=center_point,
    width=width,
    height=height,
    angle=angle,
    color="skyblue",  # fill color
    alpha=0.6,  # transparency
)

ax.add_patch(ellipse)

buffer = patches.Ellipse(
    xy=center_point,
    width=width + 10000,
    height=height + 10000,
    angle=angle,
    color="skyblue",  # fill color
    alpha=0.6,  # transparency
    linestyle="--",
    fill=None,
)

ax.add_patch(buffer)

# convex hull of all vent locations
hull_points = vents_all[["easting", "northing"]].to_numpy()
hull = ConvexHull(hull_points)
hull_vertices = hull_points[np.append(hull.vertices, hull.vertices[0])]
ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], color="red", linewidth=1, zorder=30)

cmap = ax.scatter(
    volume.easting,
    volume.northing,
    c=volume["Hopkins2020_Age"],
    # s=volume["volume"],
    cmap="magma_r",
    marker="^",
    # norm=mcolors.LogNorm(),
    # edgecolors="w",
    linewidth=1,
)

cax = fig.add_axes([0.93, 0.15, 0.025, 0.4])

cbar = fig.colorbar(cmap, label="Age [ka]", cax=cax)
cbar.ax.invert_yaxis()

ax.scatter(center_point[0], center_point[1], c="k", s=100)

ax.scatter(no_age.easting, no_age.northing, facecolors="none", edgecolors="b")

for _, row in vents.iterrows():
    ax.annotate(
        "{:.0f}".format(row["age_rank"]),
        (row["easting"], row["northing"]),
        textcoords="offset points",
        xytext=(5, -2),
        fontsize=8,
        zorder=40,
    )

ax.set_xlim(1740000, 1785000)
ax.set_ylim(5895000, 5935000)
ax.set_aspect("equal")
ax.set_ylabel("Northing NZTM [m]")
ax.set_xlabel("Easting NZTM [m]")
ax.tick_params(which="both", top=True, right=True, labeltop=False, labelright=False)

# name labels connected to each vent by a thin leader line, stacked vertically
# in the quadrant (relative to center_point) the vent falls in
xlim = ax.get_xlim()
ylim = ax.get_ylim()
margin_x_left = (xlim[1] - xlim[0]) * 0.19
margin_x_right = (xlim[1] - xlim[0]) * 0.21
pad_y = (ylim[1] - ylim[0]) * 0.03
label_x = {"left": xlim[0] + margin_x_left, "right": xlim[1] - margin_x_right}

quadrants = {"top_left": [], "top_right": [], "bottom_left": [], "bottom_right": []}
for _, row in vents_all.iterrows():
    side = "right" if row["easting"] >= center_point[0] else "left"
    vert = "top" if row["northing"] >= center_point[1] else "bottom"
    quadrants[f"{vert}_{side}"].append(row)

for key, rows in quadrants.items():
    vert, side = key.split("_")
    rows = sorted(rows, key=lambda r: r["northing"])
    if vert == "top":
        y_positions = np.linspace(center_point[1] + pad_y, ylim[1] - pad_y, len(rows))
    else:
        y_positions = np.linspace(ylim[0] + pad_y, center_point[1] - pad_y, len(rows))
    ha = "left" if side == "right" else "right"
    for row, y_pos in zip(rows, y_positions):
        if pd.notna(row["Reo"]) and str(row["Reo"]).strip():
            label_text = f"{row['Reo']}/{row['name']}"
        else:
            label_text = row["name"]
        if pd.notna(row["Hopkins2020_Age"]):
            label_text += f" [{row['Hopkins2020_Age']:g}]"
        ax.annotate(
            label_text,
            xy=(row["easting"], row["northing"]),
            xytext=(label_x[side], y_pos),
            fontsize=5,
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, shrinkA=0, shrinkB=2),
            zorder=10035,
        )

# inset map showing New Zealand outline with the main map extent
main_xlim = ax.get_xlim()
main_ylim = ax.get_ylim()

inset_ax = fig.add_axes([0.82, 0.61, 0.22, 0.22])
coast.plot(ax=inset_ax, linewidth=0.3, zorder=0, color="gray", edgecolor="k")

extent_box = patches.Rectangle(
    (main_xlim[0], main_ylim[0]),
    main_xlim[1] - main_xlim[0],
    main_ylim[1] - main_ylim[0],
    linewidth=0.8,
    edgecolor="red",
    facecolor="red",
    alpha=0.5,
)
inset_ax.add_patch(extent_box)

inset_ax.set_aspect("equal")
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.patch.set_alpha(0.5)
for spine in inset_ax.spines.values():
    spine.set_linewidth(0.6)
    # spine.set_alpha(0.5)

plt.savefig(
    "D:/dropbox/AVF/paper/revised_version/Figures/Figure1_AVF_vent_location_map_no_scaling.png",
    dpi=300,
)
plt.savefig(
    "D:/dropbox/AVF/paper/revised_version/Figures/Figure1_AVF_vent_location_map_no_scaling.svg",
    dpi=300,
)
