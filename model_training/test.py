import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import DogBehaviorDataset
from model_factory import get_model
import config

def evaluate_model(model, test_loader, device="cpu", label_names=None):
    model = model.to(device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().permute(0, 2, 1).to(device)
            y = y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    # Accuracy
    acc = accuracy_score(trues, preds)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Classification report
    print("\n📋 Classification Report:")
    all_labels = np.arange(len(label_names))
    print(classification_report(
        trues, preds,
        labels=all_labels,
        target_names=label_names,
        digits=4,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(trues, preds, labels=all_labels)
    plot_confusion_matrix(cm, label_names)


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("✅ Confusion matrix saved to 'confusion_matrix.png'.")

if __name__ == "__main__":
    dataset = DogBehaviorDataset(config.DATA_PATH, window_size=config.WINDOW_SIZE, step=config.STEP_SIZE)
    num_classes = len(np.unique(dataset.Y))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_ds = random_split(dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    model = get_model(config.MODEL_NAME, input_channels=6, num_classes=num_classes)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"🖥️ Using device: {device}")

    # 取出分类名字（反向解码）
    label_names = dataset.label_encoder.classes_.tolist()

    evaluate_model(model, test_loader, device=device, label_names=label_names)
