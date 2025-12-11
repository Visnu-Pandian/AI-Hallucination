import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =========================================
# SECTION 0: LOAD DATA
# =========================================

with open("prompt_complexities.json", encoding="utf-8") as f:
    complexities_data = json.load(f)

with open("hallucination_ratings.json", encoding="utf-8") as f:
    ratings_data = json.load(f)

complexities = complexities_data["complexities"]
ratings = ratings_data["ratings"]

# Validate counts match
assert len(complexities) == len(ratings), (
    f"Mismatch: {len(complexities)} complexities vs {len(ratings)} ratings"
)

print(f"Loaded {len(complexities)} data points")

# =========================================
# SECTION 1: EXTRACT PAIRWISE VALUES
# =========================================

complexity_values = [entry["complexity"] for entry in complexities]
severity_values = [entry["severity"] for entry in ratings]

# Filter out any fallback ratings (severity = -1)
paired_data = [
    (c, s) for c, s in zip(complexity_values, severity_values) if s >= 0
]

filtered_complexity = np.array([p[0] for p in paired_data])
filtered_severity = np.array([p[1] for p in paired_data])

print(f"Valid pairs after filtering: {len(paired_data)}")

# =========================================
# SECTION 2: CALCULATE CORRELATION
# =========================================

# Calculate Pearson correlation using numpy (cleaner typing)
correlation_matrix = np.corrcoef(filtered_complexity, filtered_severity)
pearson_r = correlation_matrix[0, 1]

print(f"\nCorrelation: r = {pearson_r:.4f}")
if abs(pearson_r) < 0.1:
    interpretation = "No correlation"
elif abs(pearson_r) < 0.3:
    interpretation = "Weak correlation"
elif abs(pearson_r) < 0.5:
    interpretation = "Moderate correlation"
else:
    interpretation = "Strong correlation"
print(f"  Interpretation: {interpretation}")

# =========================================
# SECTION 3: SIMPLE VISUALIZATION
# =========================================

# Calculate average severity for each complexity bin
complexity_bins = np.arange(0, 11, 1)  # 0-1, 1-2, ..., 9-10
bin_centers = []
avg_severity = []
std_severity = []

for i in range(len(complexity_bins) - 1):
    low, high = complexity_bins[i], complexity_bins[i + 1]
    mask = (filtered_complexity >= low) & (filtered_complexity < high)
    if mask.sum() > 0:
        bin_centers.append((low + high) / 2)
        avg_severity.append(filtered_severity[mask].mean())
        std_severity.append(filtered_severity[mask].std())

bin_centers = np.array(bin_centers)
avg_severity = np.array(avg_severity)
std_severity = np.array(std_severity)

# Create simple bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(bin_centers, avg_severity, width=0.8, color='steelblue', 
              edgecolor='white', alpha=0.8, yerr=std_severity, capsize=4)

# Add trend line
slope, intercept, _, _, _ = stats.linregress(bin_centers, avg_severity)
trend_x = np.array([bin_centers.min(), bin_centers.max()])
trend_y = slope * trend_x + intercept
ax.plot(trend_x, trend_y, 'r--', linewidth=2, label=f'Trend (r = {pearson_r:.3f})')

ax.set_xlabel("Prompt Complexity Score", fontsize=12)
ax.set_ylabel("Average Hallucination Severity", fontsize=12)
ax.set_title("Does Prompt Complexity Affect Hallucination Severity?", fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 11))
ax.set_ylim(0, 10)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add correlation annotation
if pearson_r > 0:
    direction = "more complex prompts -> slightly higher severity"
else:
    direction = "more complex prompts -> slightly lower severity"

ax.text(0.02, 0.98, f"Correlation: r = {pearson_r:.3f}\nInterpretation: {interpretation}\n({direction})", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("correlation_plot.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nDONE - Plot saved to correlation_plot.png")
