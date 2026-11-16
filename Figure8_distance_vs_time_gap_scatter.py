# -*- coding: utf-8 -*-
"""
Panel A: scatter of vent-to-vent migration distance vs. the time gap (repose
interval) since the previous eruption. Points are colored by eruption order
and annotated with their order number for identification, with diamond
markers flagging eruptions below the 95% significance threshold.
Panel B: azimuth rose diagram of vent-pair orientations (as in
Figure8_Structural_plot.py).

@author: craigm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# SET GLOBAL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

datafile = "D:/Dropbox/AVF/data/AVF structural analysis.xlsx"
data = pd.read_excel(datafile)

# Sort oldest to youngest (order 1 = oldest) and compute time gap to previous eruption
data = data.sort_values("order").reset_index(drop=True)
data["time_gap_ka"] = data["Hopkins2020_Age"].shift(1) - data["Hopkins2020_Age"]

# Drop the first eruption (no previous vent, so no distance/time gap defined)
valid = data.dropna(subset=["distance", "time_gap_ka"]).copy()
valid["distance_km"] = valid["distance"] / 1000.0
is_sig = valid["95% threshold?"].astype(str).str.upper() == "Y"

print(
    "Eruption order numbers below the 95% threshold (red outline):",
    valid.loc[is_sig, "order"].astype(int).tolist(),
)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5.5))

ax = fig.add_subplot(121)

sc = ax.scatter(
    valid.loc[~is_sig, "time_gap_ka"],
    valid.loc[~is_sig, "distance_km"],
    c=valid.loc[~is_sig, "order"],
    cmap="viridis",
    marker="o",
    edgecolor="k",
    s=200,
    zorder=3,
    alpha=0.8,
    label="Outside spatial cluster threshold",
)
ax.scatter(
    valid.loc[is_sig, "time_gap_ka"],
    valid.loc[is_sig, "distance_km"],
    c=valid.loc[is_sig, "order"],
    cmap="viridis",
    vmin=valid["order"].min(),
    vmax=valid["order"].max(),
    marker="D",
    edgecolor="k",
    linewidth=1.5,
    s=100,
    zorder=4,
    label="Inside spatial cluster threshold",
)

# Label non-clustered points with the eruption number centred inside the marker
for _, row in valid.loc[~is_sig].iterrows():
    ax.text(
        row["time_gap_ka"],
        row["distance_km"],
        int(row["order"]),
        fontsize=6,
        color="white",
        ha="center",
        va="center",
        zorder=5,
        path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
    )

# Orders 3 and 16 sit far enough from the dense cluster to label inside the marker
inside_label_orders = [3, 16]
sig_all = valid.loc[is_sig]
for _, row in sig_all.loc[sig_all["order"].isin(inside_label_orders)].iterrows():
    ax.text(
        row["time_gap_ka"],
        row["distance_km"],
        int(row["order"]),
        fontsize=6,
        color="white",
        ha="center",
        va="center",
        zorder=5,
        path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
    )

# The remaining significant points cluster tightly together (small distance &
# time gap), so align their labels vertically at a fixed x position with
# leader lines back to their actual marker
sig_rows = (
    sig_all.loc[~sig_all["order"].isin(inside_label_orders)]
    .sort_values("distance_km")
    .reset_index(drop=True)
)
n_sig = len(sig_rows)
label_x = 2.5
label_ys = np.linspace(
    sig_rows["distance_km"].min() - 0.75, sig_rows["distance_km"].max() + 1.5, n_sig
)
for i, row in sig_rows.iterrows():
    ax.annotate(
        int(row["order"]),
        (row["time_gap_ka"], row["distance_km"]),
        xytext=(label_x, label_ys[i]),
        textcoords="data",
        fontsize=6,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
    )

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Eruption order (oldest \u2192 youngest)")

ax.axhline(2584 / 1000.0, color="red", linewidth=1.2, zorder=2)

ax.set_xlabel("Time gap since previous eruption [ka]")
ax.set_ylabel("Migration distance from previous eruption [km]")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(
    loc="upper right",
    framealpha=0.9,
    borderpad=1.2,
    labelspacing=1.0,
    handletextpad=1.0,
)
ax.text(
    0.02,
    0.98,
    "A",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="normal",
    verticalalignment="top",
    horizontalalignment="left",
)

# ── Panel B: azimuth rose diagram (as in Figure8_Structural_plot.py) ─────────
azi_data = pd.read_excel(datafile, sheet_name="Final azimuths")

ax2 = fig.add_subplot(122, projection="polar")
num_bins = 36
bin_edges = np.linspace(0, 360, num_bins + 1)
bin_width = 360 / num_bins

counts, bin_edges = np.histogram(azi_data['double'], bins=bin_edges)
theta = np.radians(bin_edges[:-1] + bin_width / 2)

ax2.bar(
    theta,
    counts,
    width=np.radians(bin_width),
    alpha=0.7,
    color="blue",
    edgecolor="navy",
)

ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.spines["polar"].set_visible(False)
ax2.set_yticks([1, 2])
ax2.text(
    0.02,
    0.98,
    "B",
    transform=ax2.transAxes,
    fontsize=14,
    fontweight="normal",
    verticalalignment="top",
    horizontalalignment="left",
)

# plot all azimuths
#theta = np.radians(bin_edges[:-1] + bin_width / 2)
#counts, bin_edges = np.histogram(azi_data['all_azimuth'], bins=bin_edges)
#
#ax2.bar(
#    theta,
#    counts,
#    width=np.radians(bin_width),
#    alpha=0.2,
#    color="gray",
#    edgecolor="navy",
#)
#
#ax2.set_theta_zero_location("N")
#ax2.set_theta_direction(-1)
#ax2.spines["polar"].set_visible(False)
#ax2.set_yticks([1, 2, 3])
#ax2.text(
#    0.02,
#    0.98,
#    "B",
#    transform=ax2.transAxes,
#    fontsize=14,
#    fontweight="normal",
#    verticalalignment="top",
#    horizontalalignment="left",
#)

plt.tight_layout()
plt.savefig(
    "D:/Dropbox/AVF/paper/revised_version/Figures/Figure8_distance_vs_time_gap_scatter.png",
    dpi=300,
)
plt.show()
