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
        audio_path = (
            os.path.splitext(os.path.join(song_dir, row["mp3_path"]))[0] + ".mp3"
        )
    elif song_dir == "../data/magnatagatune/wav/":
        audio_path = (
            os.path.splitext(os.path.join(song_dir, row["mp3_path"]))[0] + ".wav"
        )
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


# def process_song(args):
#     mp3_path, label, total_len, win_len, step_len = args
#     y, sr = librosa.load(mp3_path)

#     padded_y = np.pad(y, (0, total_len * sr - len(y)), "constant")
#     num_segments = (len(padded_y) - win_len * sr) // (step_len * sr) + 1

#     log_mel_list = []
#     label_list = []
#     for _ in range(num_segments):
#         start = _ * step_len * sr
#         end = start + win_len * sr
#         segment = padded_y[start:end]

#         log_mel_spec = extract_log_mel(segment, sr)
#         log_mel_list.append(log_mel_spec)
#         label_list.append(label)
#     return log_mel_list, label_list


# def process_directory(
#     data_dir, save_dir, binary_data, song_dir, n_workers, total_len, win_len, step_len
# ):
#     data = np.load(data_dir)
#     total_length = len(data)

#     with multiprocessing.Pool(n_workers) as pool:
#         log_mel_list = []
#         label_list = []
#         mp3_paths = []
#         labels = []

#         for mp3_path in data:
#             index, mp3_path = mp3_path.split("\t")
#             mp3_path = os.path.join(song_dir, mp3_path)
#             mp3_paths.append(mp3_path)
#             labels.append(binary_data[int(index)])

#         args = [
#             (mp3_path, label, total_len, win_len, step_len)
#             for mp3_path, label in zip(mp3_paths, labels)
#         ]

#         for result in tqdm(
#             pool.imap(process_song, args),
#             total=total_length,
#             desc=f"Processing {data_dir}",
#         ):
#             log_mel, label = result
#             log_mel_list.extend(log_mel)
#             label_list.extend(label)

#     with h5py.File(os.path.join(save_dir, "log_mel.h5"), "w") as hf:
#         hf.create_dataset("log_mel", data=np.array(log_mel_list))

#     with h5py.File(os.path.join(save_dir, "label.h5"), "w") as hf:
#         hf.create_dataset("label", data=np.array(label_list))


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
                os.path.join(save_dir, "log_mel", f"log_mel_{row_idx}_{seg_idx}.npy"),
                log_mel_spec,
            )
        elif feature_type == "mfcc":
            mfcc = extract_mfcc(segment, sr)
            np.save(
                os.path.join(save_dir, "mfcc", f"mfcc_{row_idx}_{seg_idx}.npy"), mfcc
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

    os.makedirs(os.path.join(save_dir, "log_mel"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "wav"), exist_ok=True)
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


# def preprocess_data_sota(
#     song_dir,
#     binary_dir,
#     tags_dir,
#     test_dir,
#     train_dir,
#     valid_dir,
#     save_base_dir,
#     n_workers=2,
#     win_len=30,
#     step_len=30,
#     total_len=30,
# ):
#     os.makedirs(os.path.join(save_base_dir, "training"), exist_ok=True)
#     os.makedirs(os.path.join(save_base_dir, "validation"), exist_ok=True)
#     os.makedirs(os.path.join(save_base_dir, "testing"), exist_ok=True)

#     def process_directory(data_dir, save_dir, binary_data, song_dir):
#         log_mel_list = []
#         label_list = []
#         data = np.load(data_dir)
#         total_length = len(data)

#         for i, mp3_path in tqdm(
#             enumerate(data), desc=f"Processing {data_dir}", total=total_length
#         ):
#             index, mp3_path = mp3_path.split("\t")
#             mp3_path = os.path.join(song_dir, mp3_path)
#             y, sr = librosa.load(mp3_path)

#             padded_y = np.pad(y, (0, total_len * sr - len(y)), "constant")
#             num_segments = (len(padded_y) - win_len * sr) // (step_len * sr) + 1

#             for _ in range(num_segments):
#                 start = _ * step_len * sr
#                 end = start + win_len * sr
#                 segment = padded_y[start:end]

#                 log_mel_spec = extract_log_mel(segment, sr)
#                 log_mel_list.append(log_mel_spec)
#                 label_list.append(binary_data[i][1:])

#         with h5py.File(os.path.join(save_dir, "log_mel.h5"), "w") as hf:
#             hf.create_dataset("log_mel", data=np.array(log_mel_list))

#         with h5py.File(os.path.join(save_dir, "label.h5"), "w") as hf:
#             hf.create_dataset("label", data=np.array(label_list))

#     binary_data = np.load(binary_dir)
#     tags_data = np.load(tags_dir)

#     process_directory(
#         train_dir, os.path.join(save_base_dir, "training"), binary_data, song_dir
#     )
#     process_directory(
#         valid_dir, os.path.join(save_base_dir, "validation"), binary_data, song_dir
#     )
#     process_directory(
#         test_dir, os.path.join(save_base_dir, "testing"), binary_data, song_dir
#     )


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
