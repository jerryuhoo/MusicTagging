import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from train.models import CRNN, HarmonicCNN, FCN, ShortChunkCNN


def load_last_model(path):
    models = [f for f in os.listdir(path) if f.endswith(".pt")]
    best_epoch = 0
    best_model_path = None
    for m in models:
        # skip best.pt
        if "best" in m:
            continue
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
        print("Using GPU")
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


def compute_confusion_matrix(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    tp = torch.sum(y_pred * y_true, dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), dim=0)
    return torch.stack([tp, fp, fn, tn], dim=1)


def log_confusion_matrix(writer, confusion_matrix, label_names, epoch):
    f1_scores = []
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, 0].item()
        fp = confusion_matrix[i, 1].item()
        fn = confusion_matrix[i, 2].item()
        tn = confusion_matrix[i, 3].item()
        # print(f"{label_names[i]}/tp:", tp)
        # print(f"{label_names[i]}/predicted positive:", tp + fp)
        # print(f"{label_names[i]}/gt positive:", tp + fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        f1_scores.append((i, f1))
        if writer is not None:
            writer.add_scalar(f"class_{i}_{label_names[i]}/precision", precision, epoch)
            writer.add_scalar(f"class_{i}_{label_names[i]}/recall", recall, epoch)
            writer.add_scalar(f"class_{i}_{label_names[i]}/f1", f1, epoch)

    sorted_f1_scores = sorted(f1_scores, key=lambda x: x[1], reverse=True)

    tp_sum = torch.sum(confusion_matrix[:, 0], dim=0)
    fp_sum = torch.sum(confusion_matrix[:, 1], dim=0)
    fn_sum = torch.sum(confusion_matrix[:, 2], dim=0)
    tn_sum = torch.sum(confusion_matrix[:, 3], dim=0)
    precision_sum = tp_sum / (tp_sum + fp_sum)
    recall_sum = tp_sum / (tp_sum + fn_sum)
    f1_sum = 2 * precision_sum * recall_sum / (precision_sum + recall_sum)

    if writer is not None:
        writer.add_scalar(f"val/precision", precision_sum, epoch)
        writer.add_scalar(f"val/recall", recall_sum, epoch)
        writer.add_scalar(f"val/f1", f1_sum, epoch)

    confusion_matrix = confusion_matrix.detach().cpu().numpy()

    # Calculate normalized confusion matrix
    positive_gt = confusion_matrix[:, 0] + confusion_matrix[:, 2]
    negative_gt = confusion_matrix[:, 1] + confusion_matrix[:, 3]
    confusion_matrix[:, 0] = confusion_matrix[:, 0] / positive_gt
    confusion_matrix[:, 2] = confusion_matrix[:, 2] / positive_gt
    confusion_matrix[:, 1] = confusion_matrix[:, 1] / negative_gt
    confusion_matrix[:, 3] = confusion_matrix[:, 3] / negative_gt
    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=(6, 6))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.colorbar()
    plt.xlabel('True label ("tp, fp, fn, tn")')
    plt.ylabel("Predicted label")
    fig2.canvas.draw()
    image = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    image = image.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    if writer is not None:
        writer.add_image("Normalized confusion matrix", image_tensor, epoch)

    print(f"precision: {precision_sum}, recall: {recall_sum}")
    print(f"f1: {f1_sum}")
    plt.clf()
    return precision_sum, recall_sum, f1_sum, sorted_f1_scores


def calculate_roc_auc(tp, fp, fn, tn):
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    thresholds, indices = torch.sort(fpr)
    tpr = tpr[indices]
    area = torch.trapz(tpr, thresholds)
    return area.item()


def get_auc(y_true, y_score):
    roc_aucs = metrics.roc_auc_score(y_true, y_score, average="macro")
    pr_aucs = metrics.average_precision_score(y_true, y_score, average="macro")
    print("roc_auc: %.4f" % roc_aucs)
    print("pr_auc: %.4f" % pr_aucs)
    return roc_aucs, pr_aucs


def plot_auc(y_true, y_score, save_path):
    plt.clf()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[idx]
    if save_path is not None:
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.savefig(os.path.join(save_path, "roc_curve.png"))
    return best_threshold


def get_model(config):
    model_name = config["model"]["name"]

    model_class_map = {
        "FCN": FCN,
        "CRNN": CRNN,
        "HarmonicCNN": HarmonicCNN,
        "ShortChunkCNN": ShortChunkCNN,
    }

    if model_name not in model_class_map:
        raise ValueError(f"Unknown model name: {model_name}")

    model_class = model_class_map[model_name]

    return model_class(
        sample_rate=config["model"]["sample_rate"],
        n_fft=config["model"]["n_fft"],
        f_min=config["model"]["f_min"],
        f_max=config["model"]["f_max"],
        n_mels=config["model"]["n_mels"],
        n_class=config["dataset"]["num_classes"],
        feature_type=config["feature_type"],
    )
