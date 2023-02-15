import librosa
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import h5py
import warnings

warnings.filterwarnings("ignore")

# Define the number of MFCCs to extract
def extract_mfcc(y, sr, n_mfcc):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_mfcc_mean(mfcc):
    mean_mean = np.mean(mfcc, axis=1)
    return mean_mean

def extract_log_mel(y, sr, n_mels=128, fmin=20, fmax=8000, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec

def preprocess_data(song_dir, csv_dir, save_dir):
    os.makedirs(os.path.join(save_dir, "mfcc"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "mfcc_mean"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "log_mel"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)
    # data segment: 30s => 5 * 10s, step length is 5s
    sr = 16000
    total_length = 30 * sr
    window_length = 10 * sr
    step = 5 * sr
    seg_count = (30 - 5) // 5

    # load csv
    df = pd.read_csv(csv_dir, sep="\t")
    label_names = df.columns.values
    for row_idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Load the audio file
        audio_path = os.path.join(song_dir, row["mp3_path"])[:-4] + ".wav"
        base_name = os.path.basename(audio_path)
        audio_name = base_name.split(".")[0]
        seg_count = 0
        try:
            x, sr = librosa.load(audio_path, sr=sr)
            # pad zero
            pad_length = 30 * sr - len(x)
            x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
            for i in range(0, len(x) - window_length + 1, step):
                # get the current segment
                end = i + window_length if i + window_length < len(x) else len(x)
                segment = x[i:end]

                mfcc = extract_mfcc(y=segment, sr=sr, n_mfcc=25)
                mfcc_mean = extract_mfcc_mean(mfcc)
                log_mel = extract_log_mel(y=segment, sr=sr)
                y = []
                for col_idx, label_name in enumerate(label_names):
                    if label_name == "mp3_path" or label_name == "features_id":
                        continue
                    # Extract label
                    label = row[col_idx]
                    y.append(label)
                np.save(os.path.join(save_dir, "label", f"label_{row_idx}_{seg_count}.npy"), np.array(y))
                np.save(os.path.join(save_dir, "mfcc", f"mfcc_{row_idx}_{seg_count}.npy"), mfcc)
                np.save(os.path.join(save_dir, "mfcc_mean", f"mfcc_mean_{row_idx}_{seg_count}.npy"), mfcc_mean)
                np.save(os.path.join(save_dir, "log_mel", f"log_mel_{row_idx}_{seg_count}.npy"), log_mel)
                seg_count = seg_count + 1
        except Exception:
            print("error: {}".format(audio_path))

    # if feats_type == "log_mel":
    #     np.save(os.path.join(save_dir, "log_mel", "log_mel.npy"), X)

    # X = np.array(X)
    # Normalize the features
    # X_mean = np.mean(X, axis=0)
    # X_var = np.std(X, axis=0)
    # X = X - X_mean / X_var
    # np.save("mean.npy", X_mean)
    # np.save("var.npy", X_var)
    # np.save(os.path.join(save_dir, "audio_features.npy"), X)

    """
    # load mean and variance from .npy files
    mean = np.load("mean.npy")
    var = np.load("var.npy")

    # use mean and variance to normalize new data
    X_new = ...
    X_new = (X_new - mean) / var
    """
    return None

def create_hdf5_dataset(data_dir, save_dir, dataset_name, batch_size):
    # Get a list of filenames sorted alphabetically
    filenames = sorted(os.listdir(data_dir))

    # Determine the shape of the data
    shape = np.load(os.path.join(data_dir, filenames[0])).shape
    dataset_shape = (len(filenames),) + shape

    # Create the HDF5 file
    with h5py.File(os.path.join(save_dir, f"{dataset_name}.h5"), "w") as f:
        dataset = f.create_dataset(dataset_name, shape=dataset_shape, dtype='float32')

        # Iterate over the files and add the data to the dataset
        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i:i+batch_size]
            batch_data = [np.load(os.path.join(data_dir, filename)) for filename in tqdm(batch_filenames)]
            dataset[i:i+len(batch_data)] = batch_data

    return dataset_shape