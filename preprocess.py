import pandas as pd
import matplotlib.pyplot as plt
from converters import perform_label_reduction
from plot_histogram import plot_histogram
from extract_feats import preprocess_data, create_hdf5_dataset
from tqdm import tqdm
import os

total_len = 30

stage = 1
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
    if not os.path.exists(save_dir + "mfcc.h5"):
        print("creating mfcc.h5")
        create_hdf5_dataset(save_dir + "mfcc", save_dir, "mfcc", 50000)
    if not os.path.exists(save_dir + "label.h5"):
        print("creating label.h5")
        create_hdf5_dataset(save_dir + "label", save_dir, "label", 50000)
    if not os.path.exists(save_dir + "log_mel.h5"):
        print("creating log_mel.h5")
        create_hdf5_dataset(save_dir + "log_mel", save_dir, "log_mel", 10000)
