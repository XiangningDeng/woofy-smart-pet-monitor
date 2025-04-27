import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import DogBehaviorDataset
from model_factory import get_model
import config

def train_model(model, train_loader, val_loader, device="cpu", epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_loader_loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training", leave=False)

        for x, y in train_loader_loop:
            x = x.float().permute(0, 2, 1).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        preds, trues = [], []
        val_loader_loop = tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Validation", leave=False)
        with torch.no_grad():
            for x, y in val_loader_loop:
                x = x.float().permute(0, 2, 1).to(device)
                y = y.to(device)
                out = model(x)
                pred = torch.argmax(out, dim=1)
                preds.extend(pred.cpu().numpy())
                trues.extend(y.cpu().numpy())

        val_acc = accuracy_score(trues, preds)
        history["val_acc"].append(val_acc)

        # ⭐️ 打印 epoch结束的 summary
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最好模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✅ Best model saved at epoch {epoch+1} with Val Acc {val_acc:.4f}")

    # 训练结束后绘制训练曲线
    plot_training_curves(history)

def plot_training_curves(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8,5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, history["train_loss"], color=color, label="Train Loss")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(epochs, history["val_acc"], color=color, label="Val Accuracy")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Validation Accuracy")
    plt.savefig("training_curves.png")
    plt.show()
    print("✅ Training curves saved to 'training_curves.png'.")

if __name__ == "__main__":
    dataset = DogBehaviorDataset(config.DATA_PATH, window_size=config.WINDOW_SIZE, step=config.STEP_SIZE)
    num_classes = len(np.unique(dataset.Y))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = get_model(config.MODEL_NAME, input_channels=6, num_classes=num_classes)


    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"🖥️ Using device: {device}")
    train_model(model, train_loader, val_loader, device=device, epochs=config.NUM_EPOCHS)
