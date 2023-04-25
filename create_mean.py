import h5py
import numpy as np

input_filepath = "preprocessed/10/training/log_mel.h5"
output_filepath = "preprocessed/10/training/log_mel_mean.h5"
chunk_size = 1000


def process_chunk(data_chunk):
    return np.mean(data_chunk, axis=-1)


with h5py.File(input_filepath, "r") as input_file:
    data = input_file["log_mel"]
    num_samples = data.shape[0]

    # Calculate the number of chunks to process
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    log_mel_mean = []

    # Process data in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk = data[start_idx:end_idx]
        log_mel_mean_chunk = process_chunk(chunk)
        log_mel_mean.append(log_mel_mean_chunk)

    # Concatenate results from all chunks
    log_mel_mean = np.concatenate(log_mel_mean, axis=0)

# Save calculated mean values to a new file
with h5py.File(output_filepath, "w") as output_file:
    output_file.create_dataset("log_mel_mean", data=log_mel_mean)

print(f"Mean values calculated and saved to '{output_filepath}'")
