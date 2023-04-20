import argparse
import torch
import numpy as np
import os
import yaml
from train.dataset import HDF5Dataset, HDF5DataLoader
from tqdm import tqdm
from utils import (
    check_device,
    compute_confusion_matrix,
    log_confusion_matrix,
    get_auc,
    get_model,
    plot_auc,
)

device = check_device()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Inference for a multi-label classification model"
)
parser.add_argument(
    "--model_folder",
    type=str,
    default="models",
    help="Path to the models folder containing sub-folders for each model",
)
args = parser.parse_args()

# Get the list of models in the models folder
model_folders = [
    os.path.join(args.model_folder, folder)
    for folder in os.listdir(args.model_folder)
    if os.path.isdir(os.path.join(args.model_folder, folder))
]

for model_folder in model_folders:
    print(f"Testing model: {os.path.basename(model_folder)}")
    # Load the config.yaml file
    config_path = os.path.join(model_folder, "config.yaml")
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Load the model checkpoint
    model_path = os.path.join(model_folder, "model_best.pt")
    checkpoint = torch.load(model_path)
    model_name = config["model"]["name"]
    loss_type = config["loss"]["type"]
    model = get_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Define the hyperparameters
    feature_length = config["feature_length"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    saved_models_count = config["saved_models_count"]
    max_models_saved = config["max_models_saved"]
    save_interval = config["save_interval"]
    num_workers = config["num_workers"]
    csv_dir = config["dataset"]["csv_dir"]

    feature_type = config["feature_type"]

    preprocessed_path = "preprocessed/" + str(feature_length) + "/"
    num_classes = 50

    if feature_type == "mfcc":
        input_dim = 25
        test_feature_h5_path = preprocessed_path + "testing" + "/mfcc.h5"
    if feature_type == "log_mel":
        input_dim = 128
        test_feature_h5_path = preprocessed_path + "testing" + "/log_mel.h5"

    test_label_h5_path = preprocessed_path + "testing" + "/label.h5"

    test_dataset = HDF5Dataset(
        feature_h5_path=test_feature_h5_path,
        label_h5_path=test_label_h5_path,
        feature_type=feature_type,
    )

    test_len = len(test_dataset)
    print("test_len", test_len)

    # Create a DataLoader
    test_loader = HDF5DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Perform inference on the input data
    with torch.no_grad():
        correct = 0
        total = 0
        confusion_matrix = torch.zeros((num_classes, 4)).to(device)
        output_array = np.empty((0, 50))
        label_array = np.empty((0, 50))
        for data in tqdm(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            outputs = model(inputs)
            total += labels.numel()
            correct += (outputs.round() == labels).sum().item()
            batch_confusion_matrix = compute_confusion_matrix(outputs, labels)
            confusion_matrix += batch_confusion_matrix
            outputs = outputs.detach().cpu().numpy()
            output_array = np.concatenate((output_array, outputs))
            label_array = np.concatenate((label_array, labels.detach().cpu().numpy()))
        acc = correct / total
        y_true = label_array.flatten()
        y_score = output_array.flatten()
        roc_auc, pr_auc = get_auc(y_true, y_score)
        precision, recall, f1 = log_confusion_matrix(None, confusion_matrix, None, None)
        best_threshold = plot_auc(y_true, y_score, model_folder)

    print("Test results")
    print("Accuracy: {:.4f}".format(acc))
    print("ROC AUC: {:.4f}".format(roc_auc))
    print("PR AUC: {:.4f}".format(pr_auc))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))
    print("Best Threshold: {:.4f}".format(best_threshold))

    # Save the best_writer_values to a text file
    with open(os.path.join(model_folder, "best_writer_values_test.txt"), "w") as f:
        f.write(f"accuracy: {acc}\n")
        f.write(f"roc_auc: {roc_auc}\n")
        f.write(f"pr_auc: {pr_auc}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")
        f.write(f"best_threshold: {best_threshold}\n")
