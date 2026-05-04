# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 08:39:06 2025

@author: craigm
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from pathlib import Path

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"
coastline = data_file = (
    Path(__file__).resolve().parent
    / "data"
    / "Coastlines_and_Islands_Polygons_50K.shp"
)


coast = gpd.read_file(coastline)

vents = pd.read_csv(data_file, comment="#")
vents_all = pd.read_csv(data_file, comment="#")

vents = vents.sort_values(by="Hopkins2020_Age")

# Ensure sorted by age (oldest → youngest)
vents = vents.sort_values("Hopkins2020_Age", ascending=True).reset_index(
    drop=True
)
vents.dropna(subset="Hopkins2020_Age", inplace=True)
vents["age_rank"] = vents["Hopkins2020_Age"].rank(
    method="min", ascending=False
)

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


fig, ax = plt.subplots(figsize=(8, 8))
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
    width=width + 5000,
    height=height + 5000,
    angle=angle,
    color="skyblue",  # fill color
    alpha=0.6,  # transparency
    linestyle="--",
    fill=None,
)

ax.add_patch(buffer)

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

cax = fig.add_axes([0.78, 0.15, 0.04, 0.4])

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

ax.set_xlim(1745000, 1780000)
ax.set_ylim(5896500, 5933500)
ax.set_aspect("equal")
ax.set_ylabel("Northing NZTM [m]")
ax.set_xlabel("Easting NZTM [m]")
plt.savefig(
    Path(__file__).resolve().parent
    / "Figure1_AVF_vent_location_map_no_scaling.png",
    dpi=300,
)
