import pandas as pd


def count_label_distribution(input_file, output_file, num_columns):
    # Read the CSV file
    df = pd.read_csv(input_file, usecols=range(num_columns), sep="\t")

    # Count the label distribution
    label_counts = df.sum(axis=0)
    
    # Sort the results by count
    sorted_label_counts = label_counts.sort_values(ascending=False)

    # Save the results to the output file
    with open(output_file, "w") as f:
        for label, count in sorted_label_counts.items():
            f.write(f"{label}: {count}\n")

    print(f"Sorted label distribution saved to {output_file}")


if __name__ == "__main__":
    input_file = "../data/magnatagatune/annotations_final_new.csv"
    output_file = "sorted_label_distribution.txt"
    num_columns = 50

    count_label_distribution(input_file, output_file, num_columns)
