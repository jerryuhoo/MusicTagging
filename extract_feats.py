import librosa
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

# Define the number of MFCCs to extract
def extract_mfcc(y, sr, n_mfcc):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_mfcc_mean(mfcc):
    mean_mean = np.mean(mfcc, axis=1)
    return mean_mean

def preprocess_data(song_dir, csv_dir, save_dir):
    os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "mfcc"), exist_ok=True)
    X = []  # mfcc
    X_mean = []  # mfcc mean
    y = []  # label

    sr = 16000
    total_length = 30 * sr
    window_length = 10 * sr
    step = 5 * sr
    seg_count = (30 - 5) // 5

    df = pd.read_csv(csv_dir, sep="\t")

    label_names = df.columns.values
    # for i in range(df.shape[0] * seg_count):
    #     y.append([])
    for row_idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Load the audio file
        audio_path = os.path.join(song_dir, row["mp3_path"])[:-4] + ".wav"
        base_name = os.path.basename(audio_path)
        audio_name = base_name.split(".")[0]
        seg_count = 0
        # segments = []
        try:
            x, sr = librosa.load(audio_path, sr=sr)
            # pad zero
            pad_length = 30 * sr - len(x)
            x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
            for i in range(0, len(x) - window_length + 1, step):
                seg_count = seg_count + 1
                # get the current segment
                end = i + window_length if i + window_length < len(x) else len(x)
                segment = x[i:end]

                # store the segment in the list
                feats = extract_mfcc(y=segment, sr=sr, n_mfcc=25)
                feats_mean = extract_mfcc_mean(feats)
                X.append(feats)
                X_mean.append(feats_mean)
        except Exception:
            print("error: {}".format(audio_path))

        for i in range(seg_count):
            y.append([])
            for col_idx, label_name in enumerate(label_names):
                if label_name == "mp3_path" or label_name == "features_id":
                    continue
                # Extract label
                label = row[col_idx]
                y[-1].append(label)

    y = np.array(y)
    X = np.array(X)
    X_mean = np.array(X_mean)
    np.save(os.path.join(save_dir, "label", "label.npy"), y)
    np.save(os.path.join(save_dir, "mfcc", "mfcc_svm.npy"), X_mean)
    np.save(os.path.join(save_dir, "mfcc", "mfcc.npy"), X)
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
    return X, y
