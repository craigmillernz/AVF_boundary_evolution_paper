"""
Monte Carlo simulation for probability that next eruption is INSIDE convex hull
with age uncertainties

Based on min_max_age_probability.py
Repeats analysis 10,000 times for testing, sampling ages from uniform distributions

Created on: April 2026
@author: craigm
"""

from shapely.geometry import Point, MultiPoint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# SET GLOBAL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

# ============================================================================
# CONFIGURATION OPTIONS
# ============================================================================
DROP_VENTS_WITH_NO_ERROR = False  # If True, drop vents with missing errors
# If False, assign small value (0.1 ka)
DEFAULT_ERROR = 5000  # Default error value if DROP_VENTS_WITH_NO_ERROR=False
n_iterations = (
    # Number of Monte Carlo iterations (10000 for testing, 100000 for final)
    10000
)
# ============================================================================

# Load data
# data_file = "D:/Dropbox/AVF/data/AVF_main_vents.csv"
data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"
vents = pd.read_csv(data_file, comment="#")

# Remove vents with missing age data
vents = vents.dropna(subset=["Hopkins2020_Age"]).reset_index(drop=True)

if DROP_VENTS_WITH_NO_ERROR:
    # Drop vents with missing error data
    n_before = len(vents)
    vents = vents.dropna(subset=["Hopkins2020_Error"]).reset_index(drop=True)
    n_dropped = n_before - len(vents)
    print(f"Dropping {n_dropped} vents with missing error data")
else:
    # Fill missing errors with a small value (essentially no uncertainty)
    vents = vents.fillna({"Hopkins2020_Error": DEFAULT_ERROR})

print(f"Using {len(vents)} vents with age and error data")

# Extract age and error information
ages = vents["Hopkins2020_Age"].values
errors = vents["Hopkins2020_Error"].values
eastings = vents["easting"].values
northings = vents["northing"].values

print(f"  - Vents with complete data: {len(ages)}")
print(f"  - Min error: {np.min(errors):.4f}, Max error: {np.max(errors):.4f}")
print(f"  - Ages range: {np.min(ages):.2f} to {np.max(ages):.2f} ka")

n_vents = len(vents)

# Store results
probability_distribution = []

print(
    f"\nRunning {n_iterations:,} Monte Carlo iterations with {
        n_vents
    } vents..."
)

# Monte Carlo loop
for iteration in range(n_iterations):
    if (iteration + 1) % 2_000 == 0:
        print(f"  Iteration {iteration + 1:,} / {n_iterations:,}")

    # Sample ages from uniform distributions for each vent
    # Uniform between (age - error) and (age + error)
    sampled_ages = np.random.uniform(
        low=ages - errors, high=ages + errors, size=n_vents
    )

    # Create index array and sort by sampled ages (oldest to youngest)
    # descending order (oldest first)
    age_indices = np.argsort(sampled_ages)[::-1]

    # Track outcomes for this iteration
    iteration_outcomes = []
    past_points = []

    # Process vents in chronological order of sampled ages
    for idx in age_indices:
        pt = Point(eastings[idx], northings[idx])

        # Need at least 3 points to define a convex hull
        if len(past_points) >= 16:
            hull = MultiPoint(past_points).convex_hull
            inside = int(hull.contains(pt) or hull.touches(pt))
            iteration_outcomes.append(inside)

        past_points.append(pt)

    # Calculate probability for this iteration
    # Probability that eruption is INSIDE the hull
    if len(iteration_outcomes) > 0:
        iteration_outcomes = np.array(iteration_outcomes)
        prob_inside = np.sum(iteration_outcomes == 1) / len(iteration_outcomes)
    else:
        prob_inside = np.nan

    probability_distribution.append(prob_inside)

probability_distribution = np.array(probability_distribution)

# Remove any NaN values
probability_distribution = probability_distribution[
    ~np.isnan(probability_distribution)
]

# Calculate statistics
print("\n" + "=" * 60)
print("RESULTS FROM MONTE CARLO SIMULATION")
print("=" * 60)
print(
    f"Number of valid iterations: {len(probability_distribution):,} / {
        n_iterations:,}"
)
print("\nProbability that next eruption is INSIDE convex hull:")
print(f"  Mean:           {np.mean(probability_distribution):.4f}")
print(f"  Median:         {np.median(probability_distribution):.4f}")
print(f"  Std Dev:        {np.std(probability_distribution):.4f}")
print(f"  Min:            {np.min(probability_distribution):.4f}")
print(f"  Max:            {np.max(probability_distribution):.4f}")
print(f"  16th percentile: {np.percentile(probability_distribution, 16):.4f}")
print(f"  84th percentile: {np.percentile(probability_distribution, 84):.4f}")
print("=" * 60)

# Save results to CSV
results_df = pd.DataFrame(
    {"probability_outside_hull": probability_distribution}
)
results_df.to_csv(
    "D:/Dropbox/AVF/data/MC_probability_distribution_young.csv", index=False
)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(probability_distribution, bins=50, edgecolor="black", alpha=0.7)
axes[0].axvline(
    np.mean(probability_distribution),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {np.mean(probability_distribution):.3f}",
)
axes[0].axvline(
    np.median(probability_distribution),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {np.median(probability_distribution):.3f}",
)
axes[0].set_xlabel("Probability of Inside Convex Hull")
axes[0].set_ylabel("Frequency")
axes[0].set_title(
    f"Distribution of Probabilities\n(from {n_iterations:,} MC iterations)"
)
axes[0].legend()
axes[0].grid(alpha=0.3)

# CDF
sorted_probs = np.sort(probability_distribution)
cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
axes[1].plot(sorted_probs, cdf, linewidth=2)
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
axes[1].axvline(
    np.median(probability_distribution),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {np.median(probability_distribution):.3f}",
)
axes[1].set_xlabel("Probability of Inside Convex Hull")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("Cumulative Distribution Function")
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(
    "D:/Dropbox/AVF/data/MC_probability_distribution_young.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
