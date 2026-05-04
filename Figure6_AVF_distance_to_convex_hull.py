# -*- coding: utf-8 -*-
"""
Compute the signed distance from each successive eruption to the convex hull
formed by all older eruptions, ordered by Hopkins2020_Age (oldest to youngest).

Negative distance = eruption is inside the existing hull
Positive distance = eruption is outside the existing hull

@author: craigm
"""

from shapely.geometry import Point, MultiPoint, Polygon, LineString
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# SET GLOBAL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"
vents = pd.read_csv(data_file, comment="#")

# Sort oldest to youngest and drop vents without ages
vents = vents.sort_values("Hopkins2020_Age", ascending=False).reset_index(
    drop=True
)
vents.dropna(subset=["Hopkins2020_Age"], inplace=True)

results = []

for i in range(len(vents)):
    row = vents.iloc[i]
    next_pt = Point(row["easting"], row["northing"])

    if i < 3:
        # Need at least 3 prior points to form a convex hull polygon
        results.append(
            {
                "name": row["name"],
                "Hopkins2020_Age": row["Hopkins2020_Age"],
                "easting": row["easting"],
                "northing": row["northing"],
                "signed_distance_m": None,
                "n_prior_vents": i,
            }
        )
        continue

    # Build hull from all older eruptions (indices 0..i-1)
    prior_points = [
        Point(vents.iloc[j]["easting"], vents.iloc[j]["northing"])
        for j in range(i)
    ]
    hull = MultiPoint(prior_points).convex_hull

    if isinstance(hull, Polygon):
        # Distance from the next eruption point to the hull boundary
        dist = hull.exterior.distance(next_pt)

        # Negative if inside, positive if outside
        if hull.contains(next_pt):
            signed_dist = -dist
        else:
            signed_dist = dist
    elif isinstance(hull, LineString):
        # Only 2 unique points or collinear — distance to the line
        signed_dist = hull.distance(next_pt)
    else:
        signed_dist = None

    results.append(
        {
            "name": row["name"],
            "Hopkins2020_Age": row["Hopkins2020_Age"],
            "easting": row["easting"],
            "northing": row["northing"],
            "signed_distance_m": signed_dist,
            "n_prior_vents": i,
        }
    )

df_results = pd.DataFrame(results)

# Convert to km for convenience
df_results["signed_distance_km"] = df_results["signed_distance_m"] / 1000.0

print(
    df_results[
        ["name", "Hopkins2020_Age", "signed_distance_m", "signed_distance_km"]
    ].to_string()
)

# Save to CSV
# output_file = "D:/Dropbox/AVF/paper/Scripts/AVF_distance_to_convex_hull.csv"
# df_results.to_csv(output_file, index=False)
# print(f"\nResults saved to {output_file}")

# %% Plot combined figure


fig, ax2 = plt.subplots(figsize=(10, 5))

valid = df_results.dropna(subset=["signed_distance_m"])
colors = ["blue" if d < 0 else "red" for d in valid["signed_distance_m"]]

legend_elements = [
    Patch(facecolor="blue", edgecolor="k", label="Inside hull"),
    Patch(facecolor="red", edgecolor="k", label="Outside hull"),
]

# Bar chart - signed distance vs age
ax2.bar(
    valid["Hopkins2020_Age"],
    valid["signed_distance_km"],
    color=colors,
    edgecolor="k",
    linewidth=0.3,
)
ax2.scatter(
    valid["Hopkins2020_Age"],
    valid["signed_distance_km"],
    color=colors,
    edgecolor="k",
    s=25,
)

ax2.axhline(0, color="k", linewidth=0.5)
ax2.set_xlabel("Age [ka]")
ax2.set_ylabel("Signed distance to convex hull [km]")
ax2.invert_xaxis()
ax2.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(
    data_file=Path(__file__).resolve().parent / "Figure6_distance_to_hull.png",
    dpi=300,
)
plt.show()
