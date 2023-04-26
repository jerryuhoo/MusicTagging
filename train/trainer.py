import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import time
from dataset import HDF5Dataset, HDF5DataLoader

# from models import CRNN, HarmonicCNN, FCN, ShortChunkCNN
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import pandas as pd
import sys

sys.path.append(".")
from utils import (
    load_last_model,
    resume_training,
    save_checkpoint,
    check_device,
    compute_confusion_matrix,
    log_confusion_matrix,
    get_auc,
    get_model,
    plot_auc,
)


device = check_device()


# parse arguments
parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument(
    "--config",
    type=str,
    default="config/config_fcn.yaml",
    help="Path to configuration file.",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Directory to save the trained model.",
)
parser.add_argument(
    "--patience",
    type=int,
    default=3,
    help="Number of epochs to wait before early stopping.",
)
args = parser.parse_args()

# Define early stopping parameters
patience = args.patience
early_stopping_counter = 0
best_val_loss = float("inf")

# load configuration file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

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
seed = config["seed"]

preprocessed_path = "preprocessed/" + str(feature_length) + "/"

if feature_type == "mfcc":
    input_dim = 25
    train_feature_h5_path = preprocessed_path + "training" + "/mfcc.h5"
    val_feature_h5_path = preprocessed_path + "validation" + "/mfcc.h5"
    test_feature_h5_path = preprocessed_path + "testing" + "/mfcc.h5"
if feature_type == "log_mel":
    input_dim = 128
    train_feature_h5_path = preprocessed_path + "training" + "/log_mel.h5"
    val_feature_h5_path = preprocessed_path + "validation" + "/log_mel.h5"
    test_feature_h5_path = preprocessed_path + "testing" + "/log_mel.h5"
if feature_type == "wav":
    input_dim = 1
    train_feature_h5_path = preprocessed_path + "training" + "/wav.h5"
    val_feature_h5_path = preprocessed_path + "validation" + "/wav.h5"
    test_feature_h5_path = preprocessed_path + "testing" + "/wav.h5"

train_label_h5_path = preprocessed_path + "training" + "/label.h5"
val_label_h5_path = preprocessed_path + "validation" + "/label.h5"
test_label_h5_path = preprocessed_path + "testing" + "/label.h5"

train_dataset = HDF5Dataset(
    feature_h5_path=train_feature_h5_path,
    label_h5_path=train_label_h5_path,
    feature_type=feature_type,
)

val_dataset = HDF5Dataset(
    feature_h5_path=val_feature_h5_path,
    label_h5_path=val_label_h5_path,
    feature_type=feature_type,
)


train_len = len(train_dataset)
val_len = len(val_dataset)
print("train_len", train_len)
print("val_len", val_len)

train_loader = HDF5DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers
)
val_loader = HDF5DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

df = pd.read_csv(csv_dir, sep="\t")
label_names = df.columns.values
print("labels", label_names)
num_classes = len(label_names) - 2
print("num_classes", num_classes)


if config["loss"]["type"] == "weightedBCE":
    # Get the targets from the dataset
    num_samples = 0
    label_count = torch.zeros(num_classes)
    for features, targets in train_loader:
        num_samples += features.size(0)
        label_count += targets.sum(dim=0)
    print("label_count", label_count)
    num_pos = label_count.sum()
    label_distribution = label_count / num_samples
    print("label_distribution", label_distribution)
    print("num_pos", num_pos)
    print("num_samples", num_samples)
    total_labels = num_samples * num_classes
    pos_weight = (total_labels - num_pos) / total_labels
    neg_weight = num_pos / total_labels
    print("pos_weight", pos_weight)
    print("neg_weight", neg_weight)
    global_class_weights = 1.0 - label_distribution
    global_class_weights = torch.tensor(
        global_class_weights, dtype=torch.float32, device=device
    )
    print("balanced_class_weights", global_class_weights)
    weighted_bce_global_class_weights_loss = nn.BCELoss(weight=global_class_weights)


# Load model
model_name = config["model"]["name"]
loss_type = config["loss"]["type"]
model = get_model(config)
model = model.to(device)
print(model)

if loss_type == "BCE":
    bce_loss = nn.BCELoss()
    loss_type2 = "BCE"
elif loss_type == "weightedBCE":
    loss_type2 = "weightedBCE_" + config["loss"]["weight"]

print("loss", loss_type2)
# Define the loss function
# criterion = nn.BCELoss()
# weights = torch.tensor([1.0, 5.0]).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")

# parse optimizer settings
optimizer_name = config["optimizer"]["name"]
optimizer_lr = config["optimizer"]["lr"]
optimizer_weight_decay = config["optimizer"]["weight_decay"]

# create optimizer
optimizer_class = getattr(optim, optimizer_name)
optimizer = optimizer_class(
    model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay
)

# Resume training
# Get the name of the config file
config_name = os.path.splitext(os.path.basename(args.config))[0]
print("config", config_name)
if args.model_path is None:
    args.model_path = os.path.join("models", config_name)

model_path = args.model_path

if not os.path.exists(model_path):
    os.makedirs(model_path)

shutil.copy2(args.config, os.path.join(model_path, "config.yaml"))

last_model_path = load_last_model(model_path)
model, optimizer, start_epoch, loss = resume_training(model, optimizer, last_model_path)

# Initialize the TensorBoard writer
writer = SummaryWriter(log_dir=model_path)
# writer_eval = SummaryWriter(log_dir=os.path.join(model_path, "eval"))


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.running_mean, 0)
        nn.init.constant_(m.running_var, 1)


torch.manual_seed(seed)
model.apply(init_weights)

# Train the model
for epoch in range(start_epoch, num_epochs):
    training_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        # optimizer.zero_grad()
        outputs = model(inputs)
        if loss_type == "weightedBCE":
            if config["loss"]["weight"] == "fixed":
                weights = torch.zeros_like(labels)
                weights[labels == 0] = 0.1  # Set weight for negative samples
                weights[labels == 1] = 0.9  # Set weight for positive samples
                weighted_bce_global_class_weights_loss = nn.BCELoss(weight=weights)
            elif config["loss"]["weight"] == "global_pn":
                weights = torch.zeros_like(labels)
                weights[labels == 0] = neg_weight
                weights[labels == 1] = pos_weight
                weighted_bce_global_class_weights_loss = nn.BCELoss(weight=weights)
            elif config["loss"]["weight"] == "batch_pn":
                weights = torch.zeros_like(labels)
                total_labels_in_batch = labels.size(0) * labels.size(1)
                num_positives = torch.sum(labels)
                num_negatives = total_labels_in_batch - num_positives
                pos_weight_batch = num_negatives / total_labels_in_batch
                neg_weight_batch = num_positives / total_labels_in_batch
                weights[labels == 0] = 1 - pos_weight_batch
                weights[labels == 1] = pos_weight_batch
                weighted_bce_global_class_weights_loss = nn.BCELoss(weight=weights)
            elif config["loss"]["weight"] == "global_class":
                weights = global_class_weights
            elif config["loss"]["weight"] == "combined1":
                weights = torch.zeros_like(labels)
                weights[labels == 0] = 1 - pos_weight
                weights[labels == 1] = pos_weight
                weights = weights * global_class_weights
                weighted_bce_global_class_weights_loss = nn.BCELoss(weight=weights)
            # elif config["loss"]["weight"] == "combined2":
            #     weights = torch.zeros_like(labels)
            #     weights[labels == 0] = 1 - pos_weight
            #     weights[labels == 1] = pos_weight
            #     weights = weights * global_class_weights
            #     weighted_bce_global_class_weights_loss = nn.BCELoss(
            #         weight=weights
            #     )
            else:
                raise ValueError("Invalid weight type")
            # loss = F.binary_cross_entropy_with_logits(outputs, labels, weight=weights)
            loss = weighted_bce_global_class_weights_loss(outputs, labels)
        elif loss_type == "BCE":
            loss = bce_loss(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss /= len(train_loader)
    writer.add_scalar(
        "train/Loss",
        training_loss,
        epoch + 1,
        walltime=time.time(),
    )

    print(f"Epoch {epoch + 1} Loss: {training_loss}")

    if (epoch + 1) % save_interval == 0:
        # validation
        with torch.no_grad():
            val_loss = 0.0
            model.eval()
            correct = 0
            total = 0
            confusion_matrix = torch.zeros((num_classes, 4)).to(device)
            output_array = torch.empty((0, 50)).to(device)
            label_array = torch.empty((0, 50)).to(device)

            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device).float()
                val_outputs = model(val_inputs)
                if loss_type == "weightedBCE":
                    if config["loss"]["weight"] == "fixed":
                        weights = torch.zeros_like(val_labels)
                        weights[
                            val_labels == 0
                        ] = 0.1  # Set weight for negative samples
                        weights[
                            val_labels == 1
                        ] = 0.9  # Set weight for positive samples
                        weighted_bce_global_class_weights_loss = nn.BCELoss(
                            weight=weights
                        )
                    elif config["loss"]["weight"] == "global_pn":
                        weights = torch.zeros_like(val_labels)
                        weights[
                            val_labels == 0
                        ] = neg_weight  # Set weight for negative samples
                        weights[
                            val_labels == 1
                        ] = pos_weight  # Set weight for positive samples
                        weighted_bce_global_class_weights_loss = nn.BCELoss(
                            weight=weights
                        )
                    elif config["loss"]["weight"] == "batch_pn":
                        weights = torch.zeros_like(val_labels)
                        total_labels_in_batch = val_labels.size(0) * val_labels.size(1)
                        num_positives = torch.sum(val_labels)
                        num_negatives = total_labels_in_batch - num_positives
                        pos_weight_batch = num_negatives / total_labels_in_batch
                        neg_weight_batch = num_positives / total_labels_in_batch
                        weights[val_labels == 0] = 1 - pos_weight_batch
                        weights[val_labels == 1] = pos_weight_batch
                        weighted_bce_global_class_weights_loss = nn.BCELoss(
                            weight=weights
                        )
                    elif config["loss"]["weight"] == "global_class":
                        weights = global_class_weights
                    elif config["loss"]["weight"] == "combined1":
                        weights = torch.zeros_like(val_labels)
                        weights[val_labels == 0] = 1 - pos_weight
                        weights[val_labels == 1] = pos_weight
                        weights = weights * global_class_weights
                        weighted_bce_global_class_weights_loss = nn.BCELoss(
                            weight=weights
                        )
                    # elif config["loss"]["weight"] == "combined2":
                    #     weights = torch.zeros_like(val_labels)
                    #     weights[val_labels == 0] = 1 - pos_weight
                    #     weights[val_labels == 1] = pos_weight
                    #     weights = weights * global_class_weights
                    #     weighted_bce_global_class_weights_loss = nn.BCELoss(
                    #         weight=weights
                    #     )
                    else:
                        raise ValueError("Invalid weight type")
                    # loss = F.binary_cross_entropy_with_logits(val_outputs, val_labels, weight=weights)
                    loss = weighted_bce_global_class_weights_loss(
                        val_outputs, val_labels
                    )

                elif loss_type == "BCE":
                    loss = bce_loss(val_outputs, val_labels)
                val_loss += loss.item()
                total += val_labels.numel()
                output_array = torch.cat((output_array, val_outputs))
                label_array = torch.cat((label_array, val_labels))
            y_true = label_array.flatten()
            y_score = output_array.flatten()
            best_threshold = plot_auc(y_true.cpu().numpy(), y_score.cpu().numpy(), None)
            roc_auc, pr_auc = get_auc(y_true.cpu().numpy(), y_score.cpu().numpy())
            confusion_matrix = compute_confusion_matrix(
                output_array, label_array, best_threshold
            )
            correct = confusion_matrix[:, 0].sum() + confusion_matrix[:, 3].sum()
            val_acc = correct / total
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
            print(f"Best Threshold: {best_threshold}")
            writer.add_scalar(
                "val/Loss",
                val_loss,
                epoch + 1,
                walltime=time.time(),
            )
            writer.add_scalar(
                "val/Accuracy",
                val_acc,
                epoch + 1,
                walltime=time.time(),
            )
            writer.add_scalar(f"val/Best_Threshold", best_threshold, epoch + 1)
            writer.add_scalar(f"val/roc_auc", roc_auc, epoch + 1)
            writer.add_scalar(f"val/pr_auc", pr_auc, epoch + 1)
            precision, recall, f1, _ = log_confusion_matrix(
                writer, confusion_matrix, label_names, epoch + 1
            )
        # save model
        if saved_models_count < max_models_saved:
            model_file = os.path.join(
                model_path, "model_epoch_{}.pt".format((epoch + 1))
            )
            save_checkpoint(model, epoch + 1, model_file, optimizer, val_loss)
            saved_models_count += 1
        else:
            # Remove the oldest saved model
            oldest_model_file = os.path.join(
                model_path,
                "model_epoch_{}.pt".format(
                    (epoch + 1) - save_interval * max_models_saved
                ),
            )
            os.remove(oldest_model_file)
            # Save the new model
            model_file = os.path.join(
                model_path, "model_epoch_{}.pt".format((epoch + 1))
            )
            save_checkpoint(model, epoch + 1, model_file, optimizer, val_loss)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            # Save best writer values
            best_writer_values = {
                "train_loss": training_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "epoch": epoch + 1,
            }

            # Save the best model
            best_model_file = os.path.join(model_path, "model_best.pt")
            save_checkpoint(model, epoch + 1, best_model_file, optimizer, val_loss)

        else:
            early_stopping_counter += 1
            print(
                f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{patience}"
            )

        if early_stopping_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break
        model.train()

# Close the TensorBoard writer
writer.close()
print("Finished training")

# Save the best_writer_values to a text file
with open(os.path.join(model_path, "best_writer_values_val.txt"), "w") as f:
    for key, value in best_writer_values.items():
        f.write(f"{key}: {value}\n")
