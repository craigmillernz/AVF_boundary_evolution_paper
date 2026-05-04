# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 08:39:06 2025

@author: craigm
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"
vents = pd.read_csv(data_file, comment="#")
vents_all = pd.read_csv(data_file, comment="#")

vents = vents.sort_values(by="Hopkins2020_Age")


# Ensure sorted by age (oldest → youngest)
# vents = vents.sort_values("Hopkins2020_Age", ascending=True).reset_index(drop=True)
vents.dropna(subset="Hopkins2020_Age", inplace=True)

vents["age_rank"] = (
    vents["Hopkins2020_Age"].rank(method="min", ascending=False).astype(int)
)

# %%
# plot
fig, ax = plt.subplots()
ax.grid(zorder=0)

vents["date_group"] = vents["Hopkins2020_ReliabilityGrouping"]

color_map = {
    "1": "green",
    "2": "orange",
    "3": "red",
    "4": "purple",
    "5": "black",
}
label_map = {
    "1": "Group 1",
    "2": "Group 2",
    "3": "Group 3",
    "4": "Group 4",
    "5": "Group 5",
}

ax.errorbar(
    vents.Hopkins2020_Age,
    vents.age_rank,
    xerr=vents.Hopkins2020_Error,
    fmt="none",
    capsize=3,
    ecolor="k",
    linewidth=1,
    zorder=29,
)

for group in sorted(vents["date_group"].unique()):
    mask = vents["date_group"] == group
    ax.scatter(
        vents.loc[mask, "Hopkins2020_Age"],
        vents.age_rank[mask],
        c=color_map[group],
        s=10,
        edgecolors="k",
        linewidths=0.5,
        zorder=30,
        label=label_map[group],
    )


rect1 = Rectangle(
    # (60, 35.5),
    (60, 0),
    width=140,
    height=17,
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1,
    zorder=19,
    alpha=0.7,
)

ax.add_patch(rect1)

rect2 = Rectangle(
    # (15, 4.5),
    (15, 17),
    width=45,
    height=32.0,
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1,
    zorder=19,
    alpha=0.7,
)

ax.add_patch(rect2)

rect3 = Rectangle(
    # (0, 0),
    (0, 49),
    width=15,
    height=4.5,
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1,
    zorder=19,
    alpha=0.7,
)

ax.add_patch(rect3)
ax.legend(
    title="Date reliability", loc="upper left", fontsize=7, title_fontsize=8
)

ax.set_xlabel("Age [ka]")
ax.set_ylabel("Eruption number")
ax.set_ylim(52, 0)
plt.savefig(
    Path(__file__).resolve().parent / "Figure2_AVF_age_vs_number.png", dpi=300
)
