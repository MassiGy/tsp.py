import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("normalized_logs_as_csv.txt")

# Multiply the distsum by -1 to reverse the y-axis
# devide by 1000 to get a scale of kilometers
df["neg_distsum"] = -1 * df["distsum(meters)"] / 1000

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df["computation_time(seconds)"], df["neg_distsum"], color='blue')

# Adding labels to each point
for i, row in df.iterrows():
    plt.text(row["computation_time(seconds)"], row["neg_distsum"] + 0.5, row["strategy"], fontsize=7, ha='center')

plt.xlabel("Computation Time (seconds)")
plt.ylabel("Negative Distsum (kilometers)")
plt.title("Computation Time vs. Distsum")
plt.grid(True)
plt.tight_layout()
plt.show()
