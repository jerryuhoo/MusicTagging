import pandas as pd
import matplotlib.pyplot as plt
from converters import perform_label_reduction
from plot_histogram import plot_histogram
from extract_feats import preprocess_data
from tqdm import tqdm
import os

stage = 1
# clean the dataset
if stage == 0:
    directory_labels = "../data/magnatagatune/"
    plot_histogram(directory_labels, "annotations_final.csv")
    perform_label_reduction(directory_labels)
    plot_histogram(directory_labels, "annotations_final_new.csv")

if stage <= 1:
    # extract mfcc and labels, save them to numpy files.
    song_dir = "../data/magnatagatune/wav/"
    csv_dir = "../data/magnatagatune/annotations_final_new.csv"
    save_dir = "preprocessed"
    os.makedirs(save_dir, exist_ok=True)
    preprocess_data(song_dir, csv_dir, save_dir)
