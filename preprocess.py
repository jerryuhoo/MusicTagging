import pandas as pd
import matplotlib.pyplot as plt
from converters import perform_label_reduction
from plot_histogram import plot_histogram
from extract_feats import preprocess_data, create_hdf5_dataset, preprocess_data_sota
from tqdm import tqdm
import os

# config
total_len = 30
delete_npy = True
stage = 2
use_sota_data = False
sample_rate = 16000
feature_type = "wav"

# clean the dataset
if stage == 0:
    directory_labels = "../data/magnatagatune/"
    plot_histogram(directory_labels, "annotations_final.csv")
    perform_label_reduction(directory_labels)
    plot_histogram(directory_labels, "annotations_final_new.csv")

song_dir = "../data/magnatagatune/mp3/"
csv_dir = "../data/magnatagatune/annotations_final_new.csv"
save_dir = "preprocessed/" + str(total_len) + "/"
os.makedirs(save_dir, exist_ok=True)

binary_dir = "../sota-music-tagging-models/split/mtat/binary.npy"
tags_dir = "../sota-music-tagging-models/split/mtat/tags.npy"
test_dir = "../sota-music-tagging-models/split/mtat/test.npy"
train_dir = "../sota-music-tagging-models/split/mtat/train.npy"
valid_dir = "../sota-music-tagging-models/split/mtat/valid.npy"


if stage <= 1:
    # extract mfcc and labels, save them to numpy files.

    if use_sota_data:
        preprocess_data_sota(
            song_dir,
            binary_dir,
            tags_dir,
            test_dir,
            train_dir,
            valid_dir,
            save_dir,
            n_workers=4,
            win_len=30,
            step_len=30,
            total_len=total_len,
            sample_rate=sample_rate,
            feature_type=feature_type,
        )
    else:
        preprocess_data(
            song_dir,
            csv_dir,
            save_dir,
            n_workers=4,
            win_len=30,
            step_len=30,
            total_len=total_len,
            sample_rate=sample_rate,
            feature_type=feature_type,
        )

if stage <= 2:
    for folder in ["training", "validation", "testing"]:
        print("processing " + folder)
        if not os.path.exists(save_dir + folder + "/mfcc.h5") and os.path.exists(
            save_dir + folder + "/mfcc"
        ):
            print("creating " + folder + "/mfcc.h5")
            create_hdf5_dataset(save_dir + folder + "/mfcc", save_dir + folder, "mfcc")
        if not os.path.exists(save_dir + folder + "/mfcc_mean.h5") and os.path.exists(
            save_dir + folder + "/mfcc_mean"
        ):
            print("creating " + folder + "/mfcc_mean.h5")
            create_hdf5_dataset(
                save_dir + folder + "/mfcc_mean", save_dir + folder, "mfcc_mean"
            )
        if not os.path.exists(save_dir + folder + "/log_mel.h5") and os.path.exists(
            save_dir + folder + "/log_mel"
        ):
            print("creating " + folder + "/log_mel.h5")
            create_hdf5_dataset(
                save_dir + folder + "/log_mel", save_dir + folder, "log_mel"
            )
        if not os.path.exists(save_dir + folder + "/wav.h5") and os.path.exists(
            save_dir + folder + "/wav"
        ):
            print("creating " + folder + "/wav.h5")
            create_hdf5_dataset(save_dir + folder + "/wav", save_dir + folder, "wav")
        if not os.path.exists(save_dir + folder + "/label.h5") and os.path.exists(
            save_dir + folder + "/label"
        ):
            print("creating " + folder + "/label.h5")
            create_hdf5_dataset(
                save_dir + folder + "/label", save_dir + folder, "label"
            )

        if delete_npy:
            # delete folders that contains npy files
            print("deleting npy files")
            for subfolder in ["mfcc", "mfcc_mean", "label", "log_mel", "wav"]:
                os.system("rm -rf " + save_dir + folder + "/" + subfolder)


print("done")
