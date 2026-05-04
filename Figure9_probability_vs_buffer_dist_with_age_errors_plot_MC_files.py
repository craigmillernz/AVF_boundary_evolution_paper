import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, MultiPoint

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 09:07:39 2025

@author: craigm
"""


# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = "D:/Dropbox/AVF/data/AVF_main_vents.csv"
vents = pd.read_csv(data_file, comment="#")

vents = vents.sort_values(by="Hopkins2020_Age")

coastline = "D:/Dropbox/GIS/NZTM/Coastlines_and_Islands_Polygons_50K.shp"
# Example DataFrame (oldest → youngest)

coast = gpd.read_file(coastline)

vents["age_max"] = vents["Hopkins2020_Age"] + vents["Hopkins2020_Error"]
vents["age_min"] = vents["Hopkins2020_Age"] - vents["Hopkins2020_Error"]

# Ensure sorted by age (oldest → youngest)
vents = vents.sort_values("Hopkins2020_Age", ascending=False).reset_index(
    drop=True
)
vents.dropna(subset="Hopkins2020_Age", inplace=True)

vents_max = vents.sort_values("age_max", ascending=False).reset_index(
    drop=True
)
vents_min = vents.sort_values("age_min", ascending=False).reset_index(
    drop=True
)

vents_max.dropna(subset="Hopkins2020_Age", inplace=True)
vents_min.dropna(subset="Hopkins2020_Age", inplace=True)
# %%
# Probability all data
prob_inside_total_young = []
prob_inside_total_all = []

prob_inside_total_all_max = []
prob_inside_total_all_min = []

prob_inside_total_young_max = []
prob_inside_total_young_min = []

buffer_dist = np.arange(0, 13000, 500)

# %%
# inside total all
for buffer in buffer_dist:
    past_points = []
    inside_count = 0
    outside_count = 0

    for idx, row in vents.iterrows():
        pt = Point(row["easting"], row["northing"])

        if len(past_points) >= 3:
            hull = MultiPoint(past_points).convex_hull

            # Add buffer
            hull_buffered = hull.buffer(buffer)

            if hull_buffered.contains(pt) or hull_buffered.touches(pt):
                inside_count += 1
            else:
                outside_count += 1

        past_points.append(pt)

    total_tested = inside_count + outside_count
    prob_inside = inside_count / total_tested if total_tested > 0 else 0
    prob_outside = outside_count / total_tested if total_tested > 0 else 0
    prob_inside_total_all = np.append(prob_inside_total_all, prob_inside)

    print(
        f"Probability of next eruption being INSIDE {buffer} m:  {
            prob_inside:.2f} all"
    )
    print(
        f"Probability of next eruption being OUTSIDE {buffer} m: {
            prob_outside:.2f} all"
    )
# %%
# inside total young
for buffer in buffer_dist:
    past_points = []
    inside_count = 0
    outside_count = 0

    for idx, row in vents.iterrows():
        pt = Point(row["easting"], row["northing"])

        if len(past_points) >= 16:
            hull = MultiPoint(past_points).convex_hull

            # Add buffer
            hull_buffered = hull.buffer(buffer)
            if hull_buffered.contains(pt) or hull_buffered.touches(pt):
                inside_count += 1
            else:
                outside_count += 1

        past_points.append(pt)

    total_tested = inside_count + outside_count
    prob_inside = inside_count / total_tested if total_tested > 0 else 0
    prob_outside = outside_count / total_tested if total_tested > 0 else 0
    prob_inside_total_young = np.append(prob_inside_total_young, prob_inside)

    print(
        f"Probability of next eruption being INSIDE {buffer} m:  {
            prob_inside:.2f} all"
    )
    print(
        f"Probability of next eruption being OUTSIDE {buffer} m: {
            prob_outside:.2f} all"
    )
# %%
# read in precalc files
buffer_5ka_all_file = "D:/Dropbox/AVF/data/MC_probability_by_buffer_distance_5ka_no_error_all_vents.csv"
buffer_5ka_young_file = "D:/Dropbox/AVF/data/MC_probability_by_buffer_distance_5ka_no_error_young_vents.csv"
buffer_drop_all_file = "D:/Dropbox/AVF/data/MC_probability_by_buffer_distance_drop_no_error_all_vents.csv"
buffer_drop_young_file = "D:/Dropbox/AVF/data/MC_probability_by_buffer_distance_drop_no_error_young_vents.csv"
leaveout_k_file = "D:/Dropbox/AVF/data/AVF withhold-k summaries.csv"
synthetic_buffer_file = "D:/Dropbox/AVF/data/Synthetic buffer (t=52).csv"
leaveout_k_young_file = "D:/dropbox/AVF/data/AVF withhold-k summaries (all vents, young results).csv"

buffer_5ka_all = pd.read_csv(buffer_5ka_all_file)
buffer_5ka_young = pd.read_csv(buffer_5ka_young_file)
buffer_drop_all = pd.read_csv(buffer_drop_all_file)
buffer_drop_young = pd.read_csv(buffer_drop_young_file)
leaveout_k = pd.read_csv(leaveout_k_file)
leaveout_k = leaveout_k[leaveout_k["target_buffer_m"] % 500 == 0]
synthetic_buffer = pd.read_csv(synthetic_buffer_file)
synthetic_buffer = synthetic_buffer[synthetic_buffer["buffer_m"] % 500 == 0]
leaveout_k_young = pd.read_csv(leaveout_k_young_file)
leaveout_k_young = leaveout_k_young[
    leaveout_k_young["target_buffer_m"] % 500 == 0
]


# %%
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    buffer_dist,
    prob_inside_total_young,
    marker=".",
    label="young AVF, <63 ka accepted date",
    linewidth=1,
)

ax.plot(
    buffer_5ka_young["buffer_distance_m"],
    buffer_5ka_young["probability_97.5_percentile"],
    marker=".",
    label="young AVF, with age uncert.",
    color="green",
)

ax.plot(
    leaveout_k_young["target_buffer_m"],
    leaveout_k_young["pct_k1"],
    marker=".",
    label="young AVF, leave 1 out",
    color="yellow",
)

ax.plot(
    buffer_dist,
    prob_inside_total_all,
    marker=".",
    label="all AVF, accepted date",
    linewidth=1,
    color="r",
)

ax.plot(
    buffer_5ka_all["buffer_distance_m"],
    buffer_5ka_all["probability_97.5_percentile"],
    marker=".",
    label="all AVF, with age uncert.",
    color="purple",
)


ax.plot(
    leaveout_k["target_buffer_m"],
    leaveout_k["pct_k1"],
    marker=".",
    label="all AVF, leave 1 out",
    color="orange",
)

ax.plot(
    synthetic_buffer["buffer_m"],
    synthetic_buffer["pct_within_hull"],
    marker=".",
    label="synthetic data for the '55th' eruption",
    color="k",
)


ax.set_xlabel("Buffer distance [m]")
ax.set_ylabel("Probability of next eruption being inside convex hull")
ax.set_ylim(0.6, 1.05)
ax.set_xlim(-100, 12500)
ax.set_xticks(np.arange(0, 12100, 1000))
ax.set_yticks(np.arange(0.6, 1.05, 0.05))
# ax.hlines(1, -100, 12000, color="k", linewidth=1)
# ax.vlines(5000, 0, 1, color="k", linewidth=1)
ax.grid(True, linestyle="--")
plt.legend(loc="best")
plt.savefig(
    "D:/dropbox/AVF/paper/Figures/Figure9_prob_vs_buffer_dist_with_errors_MC",
    dpi=300,
)
