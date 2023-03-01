import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from torch.utils.data import DataLoader, random_split
import os
import time
from dataset import HDF5Dataset, HDF5DataLoader
from models import CRNN, HarmonicCNN, FCN
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append(".")
from utils import (
    load_best_model,
    resume_training,
    save_checkpoint,
    check_device,
    compute_confusion_matrix,
    log_confusion_matrix,
    get_auc
)


device = check_device()

# Define the hyperparameters
num_classes = 50
batch_size = 32
num_epochs = 50
saved_models_count = 0
max_models_saved = 5
save_interval = 5
learning_rate = 0.0001
num_workers = 8
feature_type = "log_mel"
if feature_type == "mfcc":
    input_dim = 25
    feature_h5_path = "preprocessed/mfcc.h5"
if feature_type == "log_mel":
    input_dim = 128
    feature_h5_path = "preprocessed/log_mel.h5"


dataset = HDF5Dataset(
    feature_h5_path=feature_h5_path,
    label_h5_path="preprocessed/label.h5",
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
# Load model
print("num_classes", num_classes)
model = FCN(
    sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=128, n_class=50
)
model = model.to(device)
print(model)

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Resume training
model_path = "models/FCN_" + str(learning_rate)
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
        loss = criterion(outputs, labels)
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
                loss = criterion(val_outputs, val_labels)
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
                label_array = np.concatenate((label_array, val_labels.detach().cpu().numpy()))
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
            log_confusion_matrix(writer, confusion_matrix, epoch + 1)
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
