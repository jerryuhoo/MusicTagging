import os
import csv

# Set the path to the models folder
models_path = "models"

# Initialize an empty dictionary to store the rows of the CSV file
csv_data = {}

# Iterate through the subdirectories in the models folder
for model_name in os.listdir(models_path):
    model_dir = os.path.join(models_path, model_name)

    # Check if the current path is a directory
    if os.path.isdir(model_dir):
        best_writer_values_path = os.path.join(model_dir, "best_writer_values_test.txt")

        # Check if the best_writer_values.txt file exists in the current model directory
        if os.path.isfile(best_writer_values_path):
            # Read the best_writer_values.txt file
            with open(best_writer_values_path, "r") as f:
                lines = f.readlines()

            # Parse the lines and create a dictionary of key-value pairs
            writer_values = {}
            for line in lines:
                key, value = line.strip().split(": ")
                writer_values[key] = round(float(value), 4)

            # Add the model name and writer values to the CSV data dictionary
            csv_data[model_name] = writer_values

# Define the header for the CSV file
header = ["Model"] + sorted(writer_values.keys())

# Sort the model names alphabetically
sorted_model_names = sorted(csv_data.keys())

# Initialize an empty list to store the rows of the CSV file
csv_rows = []

# Iterate through the sorted model names and add the model name and writer values to the CSV rows
for model_name in sorted_model_names:
    row = [model_name] + [csv_data[model_name][key] for key in sorted(writer_values)]
    csv_rows.append(row)

# Save the CSV data to a file
csv_file_path = os.path.join(models_path, "summary.csv")
with open(csv_file_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)
    csv_writer.writerows(csv_rows)

print(f"Summary CSV saved to: {csv_file_path}")
