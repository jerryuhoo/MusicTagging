import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
from dataset import MusicTaggingDataset
from cnn import CNN
from tqdm import tqdm
import tensorboardX


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


if torch.cuda.is_available():
    print("Using GPU for training.")
    device = torch.device("cuda")
else:
    print("Using CPU for training.")
    device = torch.device("cpu")


# load CNN model
model = CNN(input_dim=25, num_classes=50)
model = model.to(device)
print(model)

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001)

model_path = "models/CNN"
if not os.path.exists(model_path):
    os.makedirs(model_path)
best_model_path = load_best_model(model, model_path)
model, optimizer, start_epoch, loss = resume_training(model, optimizer, best_model_path)


# Load data
mfcc_folder = "preprocessed/mfcc"
label_folder = "preprocessed/label"

# a list of MFCC features, each of shape (num_frames, num_mfcc_coeffs)
mfcc_features = np.load(os.path.join(mfcc_folder, "mfcc.npy"), allow_pickle=True)
print("mfcc_features.shape", mfcc_features.shape)

# a list of multi-labels, each of shape (num_labels,)
multi_labels = np.load(os.path.join(label_folder, "label.npy"), allow_pickle=True)
print("multi_labels.shape", multi_labels.shape)


# Combine the MFCC features and multi-labels into a list of tuples
data_list = [(mfcc_features[i], multi_labels[i]) for i in range(len(mfcc_features))]
train_size = int(0.7 * len(data_list))
val_size = int(0.2 * len(data_list))
test_size = len(data_list) - train_size - val_size
data_list_train, data_list_val, data_list_test = torch.utils.data.random_split(
    data_list, [train_size, val_size, test_size]
)
print("train:", len(data_list_train))
print("val:", len(data_list_val))
print("test:", len(data_list_test))
# Load dataset
dataset_train = MusicTaggingDataset(data_list_train)
dataset_val = MusicTaggingDataset(data_list_val)
dataset_test = MusicTaggingDataset(data_list_test)

# Initialize the TensorBoard writer
writer = tensorboardX.SummaryWriter(model_path)

# Define the batch size
batch_size = 32

# Define the data loader
train_loader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True, num_workers=2
)

# Define the number of training epochs
num_epochs = 1000
saved_models_count = 0
max_models_saved = 5
save_interval = 5

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
    writer.add_scalar("Training Loss", training_loss, epoch + 1)
    writer.add_scalar("Training Accuracy", training_acc, epoch + 1)
    print(f"Epoch {epoch + 1} loss: {training_loss}, Train Accuracy: {training_acc}")

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
            writer.add_scalar("Validation Loss", val_loss, epoch + 1)
            writer.add_scalar("Validation Accuracy", val_acc, epoch + 1)
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
