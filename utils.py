import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool


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


def save_checkpoint(model, epoch, model_path, optimizer, loss):
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
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU for training.")
        device = torch.device("cpu")
    return device


def compute_confusion_matrix(y_pred, y_true):
    y_pred = y_pred.round()
    tp = (y_pred * y_true).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
    # return torch.stack([tp, fp, fn, tn], dim=1)
    return torch.stack([tp, fp, fn, tn], dim=1)


def log_confusion_matrix(writer, confusion_matrix, epoch):
    # for i in range(confusion_matrix.shape[0]):
    #     tp = confusion_matrix[i, 0]
    #     fp = confusion_matrix[i, 1]
    #     fn = confusion_matrix[i, 2]
    #     tn = confusion_matrix[i, 3]
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     f1 = 2 * precision * recall / (precision + recall)
    #     writer.add_scalar(f'class_{i}/precision', precision, epoch)
    #     writer.add_scalar(f'class_{i}/recall', recall, epoch)
    #     writer.add_scalar(f'class_{i}/f1', f1, epoch)
    confusion_matrix = confusion_matrix.detach().cpu().numpy()
    # print("confusion_matrix",confusion_matrix)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.colorbar()
    plt.xlabel('True label ("tp, fp, fn, tn")')
    plt.ylabel("Predicted label")
    fig.canvas.draw()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    writer.add_image("Confusion matrix", image_tensor, epoch)