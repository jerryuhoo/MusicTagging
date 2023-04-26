import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
filename = "../data/magnatagatune/annotations_final_new.csv"
data = pd.read_csv(filename, sep="\t")

# Select the first 50 columns representing labels
label_columns = data.columns[:50]
labels = data[label_columns]

# Calculate the number of labels in each sample
num_labels_per_sample = labels.sum(axis=1)

# Calculate the average number of labels per sample
average_num_labels = num_labels_per_sample.mean()
print(f"Average number of labels per sample: {average_num_labels:.2f}")

# Plot the histogram
max_labels = num_labels_per_sample.max()
bins = np.arange(0.5, max_labels + 1.5, 1)
plt.hist(num_labels_per_sample, bins=bins, edgecolor="k")
plt.xlabel("Number of Labels per Sample")
plt.ylabel("Count")
plt.title("Histogram of Labels per Sample")
plt.xticks(np.arange(1, max_labels + 2, step=max(1, max_labels // 5)))
plt.grid(axis="y", alpha=0.75)
plt.savefig("Avg Tag Histogram.png")
