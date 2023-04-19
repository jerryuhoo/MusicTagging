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


def process_song(args):
    (
        row_idx,
        mp3_path,
        label,
        total_len,
        win_len,
        step_len,
        save_dir,
        sr,
        feature_type,
    ) = args
    try:
        y, _ = librosa.load(mp3_path, sr=sr)
        padded_y = np.pad(y, (0, total_len * sr - len(y)), "constant")
        num_segments = (len(padded_y) - win_len * sr) // (step_len * sr) + 1

        for seg_idx in range(num_segments):
            start = seg_idx * step_len * sr
            end = start + win_len * sr
            segment = padded_y[start:end]
            if feature_type == "log_mel":
                log_mel_spec = extract_log_mel(segment, sr)
                np.save(
                    os.path.join(
                        save_dir, "log_mel", f"log_mel_{row_idx}_{seg_idx}.npy"
                    ),
                    log_mel_spec,
                )
            elif feature_type == "mfcc":
                mfcc = extract_mfcc(segment, sr)
                np.save(
                    os.path.join(save_dir, "mfcc", f"mfcc_{row_idx}_{seg_idx}.npy"),
                    mfcc,
                )
            elif feature_type == "mfcc_mean":
                mfcc_mean = extract_mfcc_mean(segment, sr)
                np.save(
                    os.path.join(
                        save_dir, "mfcc_mean", f"mfcc_mean_{row_idx}_{seg_idx}.npy"
                    ),
                    mfcc_mean,
                )
            elif feature_type == "wav":
                np.save(
                    os.path.join(save_dir, "wav", f"wav_{row_idx}_{seg_idx}.npy"),
                    segment,
                )
            np.save(
                os.path.join(save_dir, "label", f"label_{row_idx}_{seg_idx}.npy"), label
            )
    except Exception:
        print("error1: {}".format(mp3_path))


def process_directory(
    data_dir,
    save_dir,
    binary_data,
    song_dir,
    n_workers,
    total_len,
    win_len,
    step_len,
    sample_rate,
    feature_type,
):
    data = np.load(data_dir)
    total_length = len(data)

    if feature_type == "mfcc":
        os.makedirs(os.path.join(save_dir, "mfcc"), exist_ok=True)
    elif feature_type == "mfcc_mean":
        os.makedirs(os.path.join(save_dir, "mfcc_mean"), exist_ok=True)
    elif feature_type == "log_mel":
        os.makedirs(os.path.join(save_dir, "log_mel"), exist_ok=True)
    elif feature_type == "wav":
        os.makedirs(os.path.join(save_dir, "wav"), exist_ok=True)
    else:
        raise ValueError("Invalid feature type")
    os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)

    with multiprocessing.Pool(n_workers) as pool:
        mp3_paths = []
        labels = []

        for mp3_path in data:
            index, mp3_path = mp3_path.split("\t")
            mp3_path = os.path.join(song_dir, mp3_path)
            mp3_paths.append(mp3_path)
            labels.append(binary_data[int(index)])

        args = [
            (
                row_idx,
                mp3_path,
                label,
                total_len,
                win_len,
                step_len,
                save_dir,
                sample_rate,
                feature_type,
            )
            for row_idx, (mp3_path, label) in enumerate(zip(mp3_paths, labels))
        ]

        list(
            tqdm(
                pool.imap(process_song, args),
                total=total_length,
                desc=f"Processing {data_dir}",
            )
        )


def preprocess_data_sota(
    song_dir,
    binary_dir,
    tags_dir,
    test_dir,
    train_dir,
    valid_dir,
    save_base_dir,
    n_workers=2,
    win_len=30,
    step_len=30,
    total_len=30,
    sample_rate=16000,
    feature_type="log_mel",
):
    os.makedirs(os.path.join(save_base_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(save_base_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(save_base_dir, "testing"), exist_ok=True)

    binary_data = np.load(binary_dir)
    tags_data = np.load(tags_dir)

    process_directory(
        train_dir,
        os.path.join(save_base_dir, "training"),
        binary_data,
        song_dir,
        n_workers,
        total_len,
        win_len,
        step_len,
        sample_rate,
        feature_type,
    )
    process_directory(
        valid_dir,
        os.path.join(save_base_dir, "validation"),
        binary_data,
        song_dir,
        n_workers,
        total_len,
        win_len,
        step_len,
        sample_rate,
        feature_type,
    )
    process_directory(
        test_dir,
        os.path.join(save_base_dir, "testing"),
        binary_data,
        song_dir,
        n_workers,
        total_len,
        win_len,
        step_len,
        sample_rate,
        feature_type,
    )


def preprocess_data(
    song_dir,
    csv_dir,
    save_base_dir,
    n_workers=2,
    win_len=10,
    step_len=5,
    total_len=30,
    sample_rate=16000,
    feature_type="log_mel",
):
    os.makedirs(os.path.join(save_base_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(save_base_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(save_base_dir, "testing"), exist_ok=True)

    total_length = total_len * sample_rate
    window_length = win_len * sample_rate
    step = step_len * sample_rate

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

            if feature_type == "mfcc":
                os.makedirs(os.path.join(save_dir, "mfcc"), exist_ok=True)
            elif feature_type == "mfcc_mean":
                os.makedirs(os.path.join(save_dir, "mfcc_mean"), exist_ok=True)
            elif feature_type == "log_mel":
                os.makedirs(os.path.join(save_dir, "log_mel"), exist_ok=True)
            elif feature_type == "wav":
                os.makedirs(os.path.join(save_dir, "wav"), exist_ok=True)
            else:
                raise ValueError("Invalid feature type")
            os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)

            audio_path = (
                os.path.splitext(os.path.join(song_dir, row["mp3_path"]))[0] + ".mp3"
            )
            label = np.array(list(row["features_id"].split("_")[0]), dtype=int)

            args_list.append(
                (
                    row_idx,
                    audio_path,
                    label,
                    total_len,
                    win_len,
                    step_len,
                    save_dir,
                    sample_rate,
                    feature_type,
                )
            )
        except Exception:
            audio_path = os.path.join(song_dir, row["mp3_path"])[:-4] + ".mp3"
            print("error2: {}".format(audio_path))

    with tqdm(total=len(args_list)) as pbar:
        for _ in pool.imap_unordered(process_song, args_list):
            pbar.update()

    pool.close()
    pool.join()

    print("done")


def create_hdf5_dataset(data_dir, save_dir, dataset_name):
    # Get a list of filenames sorted alphabetically
    filenames = sorted(os.listdir(data_dir))

    # Determine the shape of the data
    shape = np.load(os.path.join(data_dir, filenames[0])).shape
    dataset_shape = (len(filenames),) + shape

    # Create the HDF5 file
    with h5py.File(os.path.join(save_dir, f"{dataset_name}.h5"), "w") as f:
        dataset = f.create_dataset(dataset_name, shape=dataset_shape, dtype="float32")

        # Iterate over the files and add the data to the dataset
        for i, filename in enumerate(tqdm(filenames)):
            data = np.load(os.path.join(data_dir, filename))
            dataset[i] = data

    return dataset_shape
