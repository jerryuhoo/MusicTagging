import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
import time
from dataset import MusicTaggingDataset
from cnn import CNN
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter


def load_best_model(model, path):
    models = [f for f in os.listdir(path) if f.endswith(".pt")]
    best_epoch = 0
    best_model_path = None
    for m in models:
        epoch = int(m.split("_")[-1].split(".")[0])
        if epoch > best_epoch:
            best_epoch = epoch
            best_model_path = os.path.join(path, m)
    return best_model_path

def resume_training(model, optimizer, model_path, epoch=0):
    if model_path is not None and os.path.isfile(model_path):
        print(f"Loading checkpoint '{model_path}'")
        checkpoint = torch.load(model_path)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss = checkpoint["loss"]
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}")
    else:
        loss = None
        print(f"No checkpoint found, Starting training from scratch")

    return model, optimizer, epoch, loss

def save_checkpoint(model, epoch, model_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )

def check_device():
    if torch.cuda.is_available():
        print("Using GPU for training.")
        device = torch.device("cuda")
    else:
        print("Using CPU for training.")
        device = torch.device("cpu")
    return device

device = check_device()

# Define the hyperparameters
num_classes = 50
batch_size = 128
num_epochs = 1000
saved_models_count = 0
max_models_saved = 5
save_interval = 5
learning_rate = 0.001
feats = "log_mel"
if feats == "mfcc":
    input_dim = 25
if feats == "log_mel":
    input_dim = 128

# Load dataset
folder_path = "preprocessed"
dataset = MusicTaggingDataset(folder_path=folder_path, feats_type=feats)

# Define the indices for the train, validation, and test sets
num_data = len(dataset)
indices = list(range(num_data))
split = [int(0.8 * num_data), int(0.9 * num_data), num_data]
train_indices, val_indices, test_indices = indices[:split[0]], indices[split[0]:split[1]], indices[split[1]:]

# Create the train, validation, and test datasets using the Subset class
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)
print("train:", len(train_dataset))
print("val:", len(val_dataset))
print("test:", len(test_dataset))

# Load CNN model
print("num_classes", num_classes)
model = CNN(input_dim=input_dim, num_classes=num_classes)
model = model.to(device)
print(model)

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Resume training
model_path = "models/CNN"
if not os.path.exists(model_path):
    os.makedirs(model_path)
best_model_path = load_best_model(model, model_path)
model, optimizer, start_epoch, loss = resume_training(model, optimizer, best_model_path)

# Initialize the TensorBoard writer
writer = SummaryWriter(log_dir=model_path)
# writer_eval = SummaryWriter(log_dir=os.path.join(model_path, "eval"))

# Define the data loader
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

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

        total += labels.size(0)
        correct += (outputs.round() == labels).sum().item()
        # writer.add_scalar(
        #     "Training Loss", training_loss, (epoch + 1) * len(train_loader) + i
        # )
    training_acc = correct / total
    training_loss /= len(train_loader)
    writer.add_scalar(
        "train/Loss",
        training_loss,
        epoch + 1,
        walltime=time.time(),
    )
    writer.add_scalar(
        "train/Accuracy",
        training_acc,
        epoch + 1,
        walltime=time.time(),
    )
    print(f"Epoch {epoch + 1} Loss: {training_loss}, Accuracy: {training_acc}")

    if (epoch + 1) % save_interval == 0:
        # validation
        with torch.no_grad():
            val_loss = 0.0
            model.eval()
            correct = 0
            total = 0
            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device).float()
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                total += val_labels.size(0)
                correct += (val_outputs.round() == val_labels).sum().item()
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
        # save model
        if saved_models_count < max_models_saved:
            model_file = os.path.join(
                model_path, "model_epoch_{}.pt".format((epoch + 1))
            )
            save_checkpoint(model, epoch + 1, model_file)
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
            save_checkpoint(model, epoch + 1, model_file)
        model.train()

# Close the TensorBoard writer
writer.close()
print("Finished training")
