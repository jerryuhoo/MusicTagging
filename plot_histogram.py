import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(directory_labels, csv_file_name):
    df = pd.read_csv(directory_labels + csv_file_name, sep="\t")
    df.info()

    # df = df.iloc[:, 1:-1]
    if csv_file_name == "annotations_final.csv":
        df = df.iloc[:, 1:-1]
    else:
        df = df.iloc[:, :-2]
    tag_counts = df.sum(axis=0).sort_values(ascending=False)
    print(tag_counts)

    # Create a bar plot
    tag_counts.plot(kind='bar')

    # Plot the histogram
    fig, ax = plt.subplots()
    ax.bar(range(len(tag_counts)), tag_counts.values)

    # Add labels and title
    plt.xlabel('Tag')
    plt.ylabel('Count')
    plt.title('Tag Histogram')
    plt.xticks()
    plt.savefig(csv_file_name[:-4] + " Tag Histogram.png")