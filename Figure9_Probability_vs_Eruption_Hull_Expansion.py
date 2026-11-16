# -*- coding: utf-8 -*-
"""
Compute the running probability of an eruption occurring outside the convex hull
of all prior eruptions, ordered by age (oldest to youngest).

Also quantifies hull expansion (delta Area and fractional growth) and computes
penalized probability gain metrics addressing reviewer feedback regarding
penalizing large hull expansions.

@author: craigm
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry import Point, MultiPoint, Polygon, LineString

# Set global font parameters
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

# Locate data file
potential_paths = [
    Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv",
    Path(__file__).resolve().parents[1] / "data" / "AVF_main_vents.csv",
    Path(__file__).resolve().parents[1] / "AVF_main_vents.csv",
    Path("D:/Dropbox/AVF/data/AVF_main_vents.csv"),
    Path("D:/Dropbox/AVF/AVF_main_vents.csv"),
]
data_file = next(
    (p for p in potential_paths if p.exists()),
    Path("D:/Dropbox/AVF/data/AVF_main_vents.csv"),
)

print(f"Reading vent data from: {data_file}")
vents = pd.read_csv(data_file, comment="#")

# Sort oldest to youngest, drop vents without ages
vents = vents.sort_values("Hopkins2020_Age", ascending=False).reset_index(drop=True)
vents.dropna(subset=["Hopkins2020_Age"], inplace=True)

n_total = len(vents)
all_pts = [
    Point(vents.iloc[j]["easting"], vents.iloc[j]["northing"]) for j in range(n_total)
]

# Calculate final total hull area for relative area normalization
final_hull = MultiPoint(all_pts).convex_hull
final_area_km2 = final_hull.area / 1e6

results = []
run_outside = 0
run_tested = 0
run_efficiency_sum = 0.0

for i in range(n_total):
    eruption_num = i + 1
    vent_name = (
        vents.iloc[i]["name"] if "name" in vents.columns else f"Vent_{eruption_num}"
    )
    age_ka = vents.iloc[i]["Hopkins2020_Age"]
    pt = all_pts[i]

    if i < 3:
        # Need at least 3 prior points for a starting convex hull polygon
        # Hull area for initial points
        current_pts = all_pts[: i + 1]
        if len(current_pts) >= 3:
            init_hull = MultiPoint(current_pts).convex_hull
            init_area = init_hull.area / 1e6 if isinstance(init_hull, Polygon) else 0.0
        else:
            init_area = 0.0

        results.append(
            {
                "eruption_number": eruption_num,
                "name": vent_name,
                "age_ka": age_ka,
                "is_outside": None,
                "signed_dist_km": None,
                "prior_hull_area_km2": None,
                "new_hull_area_km2": init_area,
                "delta_area_km2": None,
                "fractional_area_growth": None,
                "pct_area_growth": None,
                "cum_outside_count": None,
                "total_tested": None,
                "prob_outside_running": None,
                "prob_inside_running": None,
                "step_efficiency_score": None,
                "cum_efficiency_score": None,
                "spatial_prob_gain": None,
            }
        )
        continue

    # Prior hull built from vents 0..i-1
    prior_hull = MultiPoint(all_pts[:i]).convex_hull
    prior_area_km2 = prior_hull.area / 1e6 if isinstance(prior_hull, Polygon) else 0.0

    # Test if current vent is outside prior hull
    if isinstance(prior_hull, Polygon):
        dist_m = prior_hull.exterior.distance(pt)
        if prior_hull.contains(pt):
            is_outside = False
            signed_dist_km = -dist_m / 1000.0
        else:
            is_outside = True
            signed_dist_km = dist_m / 1000.0
    elif isinstance(prior_hull, LineString):
        is_outside = prior_hull.distance(pt) > 1e-6
        signed_dist_km = prior_hull.distance(pt) / 1000.0
    else:
        continue

    # New hull including current vent
    new_hull = MultiPoint(all_pts[: i + 1]).convex_hull
    new_area_km2 = new_hull.area / 1e6 if isinstance(new_hull, Polygon) else 0.0

    # Hull expansion metrics
    delta_area_km2 = max(0.0, new_area_km2 - prior_area_km2)
    fractional_growth = delta_area_km2 / prior_area_km2 if prior_area_km2 > 0 else 0.0
    pct_growth = fractional_growth * 100.0

    # Cumulative running probability
    run_tested += 1
    if is_outside:
        run_outside += 1

    prob_outside = run_outside / run_tested
    prob_inside = 1.0 - prob_outside

    # Reviewer's penalty quantification:
    # 1. Step efficiency: 1 / (1 + delta_A / A_prev)
    #    Score is 1.0 if inside (no expansion needed), and diminishes towards 0 as expansion increases.
    step_eff = 1.0 / (1.0 + fractional_growth)
    run_efficiency_sum += step_eff
    cum_eff = run_efficiency_sum / run_tested

    # 2. Spatial probability gain = P_inside / (A_prior / A_final)
    #    Measures probability gain over spatial random chance given prior hull area
    rel_area = prior_area_km2 / final_area_km2 if final_area_km2 > 0 else 1.0
    prob_gain = (prob_inside / rel_area) if rel_area > 0 else 0.0

    # 3. Combined score = per-step efficiency weighted by running P(inside), stays in [0, 1]
    combined_score = step_eff * prob_inside

    results.append(
        {
            "eruption_number": eruption_num,
            "name": vent_name,
            "age_ka": age_ka,
            "is_outside": is_outside,
            "signed_dist_km": signed_dist_km,
            "prior_hull_area_km2": prior_area_km2,
            "new_hull_area_km2": new_area_km2,
            "delta_area_km2": delta_area_km2,
            "fractional_area_growth": fractional_growth,
            "pct_area_growth": pct_growth,
            "cum_outside_count": run_outside,
            "total_tested": run_tested,
            "prob_outside_running": prob_outside,
            "prob_inside_running": prob_inside,
            "step_efficiency_score": step_eff,
            "cum_efficiency_score": cum_eff,
            "spatial_prob_gain": prob_gain,
            "combined_score": combined_score,
        }
    )

df = pd.DataFrame(results)

# Save results to CSV
output_csv = Path(__file__).resolve().parent / "AVF_probability_and_hull_expansion.csv"
df.to_csv(output_csv, index=False, float_format="%.4f")
print(f"Results saved to: {output_csv}")

# Print summary table of tested eruptions
tested_df = df.dropna(subset=["is_outside"]).copy()
print("\n--- Summary of Tested Eruptions (from Eruption 4 onward) ---")
print(
    tested_df[
        [
            "eruption_number",
            "name",
            "age_ka",
            "is_outside",
            "prior_hull_area_km2",
            "delta_area_km2",
            "pct_area_growth",
            "prob_outside_running",
            "cum_efficiency_score",
            "spatial_prob_gain",
            "combined_score",
        ]
    ].to_string(index=False)
)

# ── Multi-Panel Plot ──────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

x = tested_df["eruption_number"].values
prob_in = tested_df["prob_inside_running"].values * 100.0  # in %
is_out = tested_df["is_outside"].values.astype(bool)

# ── Panel 1: Running Probability Inside Convex Hull ──────────────────────────
ax1.plot(
    x,
    prob_in,
    color="#1F78B4",
    linewidth=2.0,
    label="Running $P_{\mathrm{inside}}$ [%]",
)
ax1.scatter(
    x[~is_out],
    prob_in[~is_out],
    color="royalblue",
    edgecolor="k",
    s=35,
    zorder=5,
    label="Inside prior hull",
)
ax1.scatter(
    x[is_out],
    prob_in[is_out],
    color="crimson",
    edgecolor="k",
    s=35,
    zorder=5,
    label="Outside prior hull",
)
loc = plticker.MultipleLocator(base=10.0)
ax1.yaxis.set_major_locator(loc)
# ax1.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
ax1.set_ylabel("Running Probability\nInside Hull [%]", fontsize=9, fontweight="normal")
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", framealpha=0.9)
ax1.text(0.02, 0.90, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="normal")

# ── Panel 2: Hull Area Growth (Delta Area and Cumulative Area) ────────────────
bar_colors = ["crimson" if out else "lightgray" for out in is_out]
ax2.bar(
    x,
    tested_df["delta_area_km2"].values,
    color=bar_colors,
    edgecolor="k",
    linewidth=0.5,
    width=0.7,
    label=r"$\Delta$ Hull Area ($\mathrm{km}^2$)",
)
ax2.set_ylabel(
    r"$\Delta$ Area ($\mathrm{km}^2$)", fontsize=9, fontweight="normal", color="crimson"
)
ax2.tick_params(axis="y", labelcolor="crimson")
ax2.grid(True, linestyle="--", alpha=0.5)

# Secondary y-axis for per-step hull efficiency score S_i
ax2_twin = ax2.twinx()
ax2_twin.plot(
    x,
    tested_df["step_efficiency_score"].values,
    color="black",
    linestyle="-",
    linewidth=1.5,
    marker="o",
    markersize=3,
    label=r"Step Efficiency [$S_i = (1+\Delta A_i/A_i)^{-1}$]",
)
ax2_twin.set_ylabel(r"$S_i$ (0 to 1)", fontsize=9, fontweight="normal", color="black")
ax2_twin.set_ylim(0, 1.05)
ax2_twin.tick_params(axis="y", labelcolor="black")

# Combine legends for Panel 2
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
ax2.legend(lines_2 + lines_2t, labels_2 + labels_2t, loc="center right", framealpha=0.9)
ax2.text(0.02, 0.90, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="normal")

# ── Panel 3: Penalized Efficiency Score & Spatial Probability Gain ───────────
ax3.plot(
    x,
    tested_df["combined_score"].values * 100,
    color="#E6AB02",
    linewidth=2.0,
    label=r"Penalised Probability [$S_i \times P_{\mathrm{inside}}$]",
)
ax3.scatter(
    x,
    # x[~is_out],
    tested_df["combined_score"].values * 100,
    color="#E6AB02",
    edgecolor="k",
    s=35,
    zorder=5,
)

ax3.plot(
    x,
    tested_df["spatial_prob_gain"].values * 100,
    color="#7570B3",
    linewidth=2.0,
    label=r"Spatial Probability Gain [$PG_i = P_{\mathrm{inside}} / (A/A_{\mathrm{final}})$]",
)
ax3.scatter(
    x,
    tested_df["spatial_prob_gain"].values * 100,
    color="#7570B3",
    edgecolor="k",
    s=35,
    zorder=5,
)

ax3.scatter(
    x,
    prob_in,
    color="grey",
    linewidth=1.5,
)

ax3.plot(
    x,
    prob_in,
    color="grey",
    linewidth=1.5,
    label=r"P$_{\mathrm{inside}}$"
)

ax3.set_ylabel(
    "Penalised Probability & \nSpatial Gain $PG_i$ [%]",
    fontsize=9,
    fontweight="normal",
    color="black",
)
ax3.set_ylim(0, 105)
ax3.tick_params(axis="y", labelcolor="black")
ax3.grid(True, linestyle="--", alpha=0.5)

ax3.legend(loc="lower right", framealpha=0.9)
ax3.text(0.02, 0.90, "(c)", transform=ax3.transAxes, fontsize=10, fontweight="normal")

ax3.set_xlabel(
    "Eruption Number (Oldest $\\rightarrow$ Youngest)", fontsize=9, fontweight="normal"
)
ax3.set_xlim(3, n_total + 1)

plt.tight_layout()

# Save figure to workspace Figures/ folder if present
fig_dir = Path(__file__).resolve().parents[1] / "Figures"
if not fig_dir.exists():
    fig_dir = Path(__file__).resolve().parent / "Figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "Figure9_Probability_vs_Eruption_Hull_Expansion.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {fig_path}")

plt.show()
