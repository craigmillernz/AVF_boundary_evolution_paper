# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:08:25 2026

@author: craigm
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

datafile = "D:/Dropbox/AVF/data/AVF structural analysis.xlsx"

data = pd.read_excel(datafile)

fig = plt.figure(figsize=(12, 5))
azi_data = pd.read_excel(datafile, sheet_name="Final azimuths", header=None)

# azi_data_flip = azi_data +180
# azi_data_all = np.column_stack((azi_data, azi_data_flip))

azi_data_all = azi_data


ax1 = fig.add_subplot(121)
ax1.plot(data.order, data.distance, marker=".", color="k", linewidth=1)
ax1.set_ylabel("Distance [m]")
ax1.set_xlabel("Eruption count [t]")

# Draw a rectangle from 0 to 2584 on y axis, spanning the width of x axis

x_min = 0
x_max = 55
rect = Rectangle(
    (x_min, 0),
    x_max - x_min,
    2584,
    linewidth=1,
    edgecolor="red",
    facecolor="red",
    alpha=0.5,
)
ax1.add_patch(rect)
# ax1.hlines(2584, x_min, x_max, linewidth=1, color="red")


# Add "A" label to top left corner
ax1.text(
    0.02,
    0.98,
    "A",
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="normal",
    verticalalignment="top",
    horizontalalignment="left",
)
ax1.set_ylim(0, 20000)
ax1.set_xlim(x_min, x_max)


ax2 = fig.add_subplot(122, projection="polar")
# Define bins for the histogram (e.g., 36 bins = 10 degree intervals)
num_bins = 36
bin_edges = np.linspace(0, 360, num_bins + 1)
bin_width = 360 / num_bins

# Create histogram
counts, bin_edges = np.histogram(azi_data_all, bins=bin_edges)

# Convert bin edges to radians for polar plot
theta = np.radians(bin_edges[:-1] + bin_width / 2)  # Center of each bin

# Create the bar plot
bars = ax2.bar(
    theta,
    counts,
    width=np.radians(bin_width),
    alpha=0.7,
    color="blue",
    edgecolor="navy",
)

ax2.set_theta_zero_location("N")  # Set 0 degrees to North
ax2.set_theta_direction(-1)  # Clockwise direction
ax2.spines["polar"].set_visible(False)

# Set radial axis to only show labels 1 and 2
ax2.set_yticks([1, 2])

# Add "B" label to top left corner
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
plt.savefig("D:\Dropbox\AVF/paper/figures/Figure8_structural.png", dpi=300)
