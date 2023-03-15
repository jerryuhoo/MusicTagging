import librosa
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import h5py
import warnings
import multiprocessing

# from memory_profiler import profile, memory_usage

warnings.filterwarnings("ignore")

# Define the number of MFCCs to extract
def extract_mfcc(y, sr, n_mfcc):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def extract_mfcc_mean(mfcc):
    mean_mean = np.mean(mfcc, axis=1)
    return mean_mean


def extract_log_mel(y, sr, n_mels=128, fmin=0, fmax=8000, n_fft=512, hop_length=256):
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


def preprocess_segment(args):
    song_dir, row_idx, row, sr, window_length, step, save_dir = args
    # print("row_idx",row_idx)
    if song_dir == "../data/magnatagatune/mp3/":
        audio_path = os.path.splitext(os.path.join(song_dir, row["mp3_path"]))[0] + ".mp3"
    elif song_dir == "../data/magnatagatune/wav/":
        audio_path = os.path.splitext(os.path.join(song_dir, row["mp3_path"]))[0] + ".wav"
    else:
        raise ValueError("song_dir should be either mp3 or wav")
    try:
        audio_data, _ = librosa.load(audio_path, sr=sr)
        pad_length = 30 * sr - len(audio_data)
        x = np.pad(audio_data, (0, pad_length), mode="constant", constant_values=0)
        segment_list = []
        for i in range(0, len(x) - window_length + 1, step):
            # get the current segment
            end = i + window_length if i + window_length < len(x) else len(x)
            segment = x[i:end]
            segment_list.append(segment)
        for seg_idx, segment in enumerate(segment_list):
            mfcc = extract_mfcc(y=segment, sr=sr, n_mfcc=25)
            mfcc_mean = extract_mfcc_mean(mfcc)
            log_mel = extract_log_mel(y=segment, sr=sr)
            y = np.array(list(row["features_id"].split("_")[0]), dtype=int)
            np.save(
                os.path.join(save_dir, "label", f"label_{row_idx}_{seg_idx}.npy"), y
            )
            np.save(
                os.path.join(save_dir, "mfcc", f"mfcc_{row_idx}_{seg_idx}.npy"), mfcc
            )
            np.save(
                os.path.join(
                    save_dir, "mfcc_mean", f"mfcc_mean_{row_idx}_{seg_idx}.npy"
                ),
                mfcc_mean,
            )
            np.save(
                os.path.join(save_dir, "log_mel", f"log_mel_{row_idx}_{seg_idx}.npy"),
                log_mel,
            )
    except Exception:
        print("error: {}".format(audio_path))


def preprocess_data(
    song_dir, csv_dir, save_base_dir, n_workers=2, win_len=10, step_len=5, total_len=30
):
    os.makedirs(os.path.join(save_base_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(save_base_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(save_base_dir, "testing"), exist_ok=True)

    # data segment: 30s => 5 * 10s, step length is 5s
    sr = 16000
    total_length = total_len * sr
    window_length = win_len * sr
    step = step_len * sr
    if step_len == 0:
        seg_count = 0
    else:
        seg_count = (total_len - win_len) // step_len + 1

    # load csv
    df = pd.read_csv(csv_dir, sep="\t")
    df = df.iloc[:, -2:]
    pool = multiprocessing.Pool(processes=n_workers)

    args_list = []
    for row_idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            mp3_path = row["mp3_path"]
            first_char = mp3_path[0]
            if first_char in "0123456789ab":
                save_dir = os.path.join(save_base_dir, "training")
            elif first_char == "c":
                save_dir = os.path.join(save_base_dir, "validation")
            elif first_char in "def":
                save_dir = os.path.join(save_base_dir, "testing")
            else:
                continue

            os.makedirs(os.path.join(save_dir, "mfcc"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "mfcc_mean"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "log_mel"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)

            args_list.append(
                (song_dir, row_idx, row, sr, window_length, step, save_dir)
            )
        except Exception:
            audio_path = os.path.join(song_dir, row["mp3_path"])[:-4] + ".mp3"
            print("error: {}".format(audio_path))

    with tqdm(total=len(args_list)) as pbar:
        for _ in pool.imap_unordered(preprocess_segment, args_list):
            pbar.update()

    # results = pool.map_async(preprocess_segment, args_list)

    pool.close()
    pool.join()

    print("done")


def create_hdf5_dataset(data_dir, save_dir, dataset_name, batch_size):
    # Get a list of filenames sorted alphabetically
    filenames = sorted(os.listdir(data_dir))

    # Determine the shape of the data
    shape = np.load(os.path.join(data_dir, filenames[0])).shape
    dataset_shape = (len(filenames),) + shape

    # Create the HDF5 file
    with h5py.File(os.path.join(save_dir, f"{dataset_name}.h5"), "w") as f:
        dataset = f.create_dataset(dataset_name, shape=dataset_shape, dtype="float32")

        # Iterate over the files and add the data to the dataset
        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i : i + batch_size]
            batch_data = [
                np.load(os.path.join(data_dir, filename))
                for filename in tqdm(batch_filenames)
            ]
            dataset[i : i + len(batch_data)] = batch_data

    return dataset_shape
