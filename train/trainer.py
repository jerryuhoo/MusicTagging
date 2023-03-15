import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
import os
import time
from dataset import HDF5Dataset, HDF5DataLoader
# from models import CRNN, HarmonicCNN, FCN, ShortChunkCNN
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import yaml
import pandas as pd
import sys

sys.path.append(".")
from utils import (
    load_best_model,
    resume_training,
    save_checkpoint,
    check_device,
    compute_confusion_matrix,
    log_confusion_matrix,
    get_auc,
    get_model
)


device = check_device()


# load configuration file
with open('config/config_fcn.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the hyperparameters
feature_length = config['feature_length']
learning_rate = config['learning_rate']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
saved_models_count = config['saved_models_count']
max_models_saved = config['max_models_saved']
save_interval = config['save_interval']
num_workers = config['num_workers']
csv_dir = config['dataset']['csv_dir']

feature_type = config['feature_type']

preprocessed_path = "preprocessed/" + str(feature_length) + "/"

if feature_type == "mfcc":
    input_dim = 25
    feature_h5_path = preprocessed_path + "/mfcc.h5"
if feature_type == "log_mel":
    input_dim = 128
    feature_h5_path = preprocessed_path + "/log_mel.h5"

label_h5_path = preprocessed_path + "/label.h5"

dataset = HDF5Dataset(
    feature_h5_path=feature_h5_path,
    label_h5_path=label_h5_path,
    feature_type=feature_type,
)

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
total_len = len(dataset)
train_len = int(total_len * train_ratio)
val_len = int(total_len * val_ratio)
test_len = total_len - train_len - val_len
print("train_len", train_len)
print("val_len", val_len)
print("test_len", test_len)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_len, val_len, test_len]
)

train_loader = HDF5DataLoader(train_dataset, batch_size=32, num_workers=num_workers)
val_loader = HDF5DataLoader(val_dataset, batch_size=32, num_workers=num_workers)

df = pd.read_csv(csv_dir, sep="\t")
label_names = df.columns.values
print("labels", label_names)
num_classes = len(label_names) - 2
print("num_classes", num_classes)


if config['loss']['type'] == "weightedBCE":
    # Get the targets from the dataset
    num_samples = 0
    label_count = torch.zeros(num_classes)
    for features, targets in train_loader:
        num_samples += features.size(0)
        label_count += targets.sum(dim=0)
    print("label_count", label_count)
    num_pos = label_count.sum()
    label_distribution = label_count / num_samples
    print("num_pos", num_pos)
    print("num_samples", num_samples)
    total_labels = num_samples * num_classes
    pos_weight = (total_labels - num_pos) / total_labels
    print("pos_weight", pos_weight)
    auto_weights = torch.tensor(1.0 - label_distribution, dtype=torch.float32, device=device)
    print("auto_weights", auto_weights)

# Load model
model_name = config['model']['name']
loss_type = config['loss']['type']
model = get_model(config)
model = model.to(device)
print(model)

if loss_type == 'BCE':
    bce_loss = nn.BCELoss()
    loss_type2 = 'BCE'
elif loss_type == 'weightedBCE':
    loss_type2 = 'weightedBCE_' + config['loss']['weight']

print("loss", loss_type2)
# Define the loss function
# criterion = nn.BCELoss()
# weights = torch.tensor([1.0, 5.0]).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")

# parse optimizer settings
optimizer_name = config['optimizer']['name']
optimizer_lr = config['optimizer']['lr']
optimizer_weight_decay = config['optimizer']['weight_decay']

# create optimizer
optimizer_class = getattr(optim, optimizer_name)
optimizer = optimizer_class(model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)

# Resume training
model_path = "models/" + model_name + "_" + loss_type2 + "_" + str(learning_rate)  + "_" + feature_type + "_" + str(feature_length)
if not os.path.exists(model_path):
    os.makedirs(model_path)
best_model_path = load_best_model(model, model_path)
model, optimizer, start_epoch, loss = resume_training(model, optimizer, best_model_path)

# Initialize the TensorBoard writer
writer = SummaryWriter(log_dir=model_path)
# writer_eval = SummaryWriter(log_dir=os.path.join(model_path, "eval"))

# Train the model
for epoch in range(start_epoch, num_epochs):
    training_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        if loss_type == 'weightedBCE':
            if config['loss']['weight'] == 'fixed':
                weights = torch.zeros_like(labels)
                weights[labels == 0] = 0.1  # Set weight for negative samples
                weights[labels == 1] = 0.9  # Set weight for positive samples
            elif config['loss']['weight'] == 'balanced_pn':
                weights = torch.zeros_like(labels)
                weights[labels == 0] = 1 - pos_weight
                weights[labels == 1] = pos_weight
            elif config['loss']['weight'] == 'balanced':
                weights = auto_weights
            loss = F.binary_cross_entropy_with_logits(outputs, labels, weight=weights)
        elif loss_type == 'BCE':
            loss = bce_loss(outputs, labels)
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
            output_array = np.empty((0, 50))
            label_array = np.empty((0, 50))

            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device).float()
                val_outputs = model(val_inputs)
                if loss_type == 'weightedBCE':
                    if config['loss']['weight'] == 'fixed':
                        weights = torch.zeros_like(val_labels)
                        weights[val_labels == 0] = 0.1  # Set weight for negative samples
                        weights[val_labels == 1] = 0.9  # Set weight for positive samples
                    elif config['loss']['weight'] == 'balanced_pn':
                        weights = torch.zeros_like(val_labels)
                        weights[val_labels == 0] = 1 - pos_weight
                        weights[val_labels == 1] = pos_weight
                    elif config['loss']['weight'] == 'balanced':
                        weights = auto_weights
                    loss = F.binary_cross_entropy_with_logits(val_outputs, val_labels, weight=weights)
                elif loss_type == 'BCE':
                    loss = bce_loss(val_outputs, val_labels)
                val_loss += loss.item()
                total += val_labels.numel()
                correct += (val_outputs.round() == val_labels).sum().item()
                batch_confusion_matrix = compute_confusion_matrix(
                    val_outputs, val_labels
                )
                confusion_matrix += batch_confusion_matrix
                val_outputs = val_outputs.detach().cpu().numpy()
                val_outputs = (val_outputs >= 0.5).astype(int)
                output_array = np.concatenate((output_array, val_outputs))
                label_array = np.concatenate(
                    (label_array, val_labels.detach().cpu().numpy())
                )
            val_acc = correct / total
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
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
            roc_auc, pr_auc = get_auc(label_array.flatten(), output_array.flatten())
            writer.add_scalar(f"val/roc_auc", roc_auc, epoch + 1)
            writer.add_scalar(f"val/pr_auc", pr_auc, epoch + 1)
            log_confusion_matrix(writer, confusion_matrix, label_names, epoch + 1)
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
        model.train()

# Close the TensorBoard writer
writer.close()
print("Finished training")
