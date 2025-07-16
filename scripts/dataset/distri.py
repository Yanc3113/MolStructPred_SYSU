import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("BBBP.csv")

# Count the labels
label_counts = df["p_np"].value_counts().sort_index()
label_ratio = df["p_np"].value_counts(normalize=True).sort_index()

# Print label distribution
print("Label distribution:")
print(label_counts)
print("\nLabel ratio:")
print(label_ratio)

# Plotting
plt.figure(figsize=(6, 4))
bars = plt.bar(
    label_counts.index.astype(str),
    label_counts.values,
    color=["lightblue", "lightsalmon"]
)

# Annotate values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# Plot styling
plt.xlabel("p_np Label (0 = Non-penetrant, 1 = Penetrant)")
plt.ylabel("Sample Count")
plt.title("Distribution of p_np Labels in BBBP Dataset created by cy")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
