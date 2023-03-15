import pandas as pd
import matplotlib.pyplot as plt
from converters import perform_label_reduction
from plot_histogram import plot_histogram
from extract_feats import preprocess_data, create_hdf5_dataset
from tqdm import tqdm
import os

# config
total_len = 30
delete_npy = True
stage = 3

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

if stage <= 1:
    # extract mfcc and labels, save them to numpy files.
    preprocess_data(
        song_dir,
        csv_dir,
        save_dir,
        n_workers=2,
        win_len=30,
        step_len=30,
        total_len=total_len,
    )

if stage <= 2:
    for folder in ["training", "validation", "testing"]:
        print("processing " + folder)
        if not os.path.exists(save_dir + folder + "/mfcc.h5"):
            print("creating " + folder + "/mfcc.h5")
            create_hdf5_dataset(
                save_dir + folder + "/mfcc", save_dir + folder, "mfcc", 50000
            )
        if not os.path.exists(save_dir + folder + "/mfcc_mean.h5"):
            print("creating " + folder + "/mfcc_mean.h5")
            create_hdf5_dataset(
                save_dir + folder + "/mfcc_mean", save_dir + folder, "mfcc_mean", 50000
            )
        if not os.path.exists(save_dir + folder + "/log_mel.h5"):
            print("creating " + folder + "/log_mel.h5")
            create_hdf5_dataset(
                save_dir + folder + "/log_mel", save_dir + folder, "log_mel", 10000
            )
        if not os.path.exists(save_dir + folder + "/label.h5"):
            print("creating " + folder + "/label.h5")
            create_hdf5_dataset(
                save_dir + folder + "/label", save_dir + folder, "label", 50000
            )

if delete_npy:
    # delete folders that contains npy files
    print("deleting npy files")
    for folder in ["training", "validation", "testing"]:
        for subfolder in ["mfcc", "mfcc_mean", "label", "log_mel"]:
            os.system("rm -rf " + save_dir + folder + "/" + subfolder)

print("done")