import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 2025
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset():
    raw_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    raw_test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    images = np.concatenate([raw_train.data, raw_test.data], axis=0)
    labels = np.array(raw_train.targets + raw_test.targets)
    return images, labels


def show_dataset_info(images, labels, classes):
    print(f"Total images  : {len(images):,}")
    print(f"Image shape   : {images[0].shape}  (H x W x C)")
    print(f"Data type     : {images.dtype}")
    print(f"Pixel range   : [{images.min()}, {images.max()}]")
    print(f"\n{'Class':>12} | {'Count':>8} | {'Proportion':>10}")
    print("-" * 36)
    for i, cls in enumerate(classes):
        cnt = int(np.sum(labels == i))
        print(f"{cls:>12} | {cnt:>8,} | {100 * cnt / len(labels):>9.1f}%")
    print("-" * 36)
    print(f"{'TOTAL':>12} | {len(labels):>8,} | {'100.0%':>10}")


def show_samples(images, labels, classes, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(images), size=10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("10 Random CIFAR-10 Samples", fontsize=13, fontweight="bold")
    for i, sample_idx in enumerate(idx):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[sample_idx])
        ax.set_title(classes[labels[sample_idx]], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("random_samples.png", dpi=100, bbox_inches="tight")
    plt.show()


def get_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def split_data(images, labels, seed):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(images, labels))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    rel_train, rel_val = next(sss2.split(images[idx_trainval], labels[idx_trainval]))
    idx_train = idx_trainval[rel_train]
    idx_val   = idx_trainval[rel_val]
    return idx_train, idx_val, idx_test


def show_split_table(labels, idx_train, idx_val, idx_test, classes):
    n = len(labels)
    print(f"Train      : {len(idx_train):6,} images ({100 * len(idx_train) / n:.1f}%)")
    print(f"Validation : {len(idx_val):6,} images ({100 * len(idx_val) / n:.1f}%)")
    print(f"Test       : {len(idx_test):6,} images ({100 * len(idx_test) / n:.1f}%)")

    def counts(arr):
        return [int(np.sum(arr == i)) for i in range(len(classes))]

    df = pd.DataFrame({
        "Class"     : classes,
        "Train"     : counts(labels[idx_train]),
        "Validation": counts(labels[idx_val]),
        "Test"      : counts(labels[idx_test]),
    })
    total_row = pd.DataFrame([{
        "Class"     : "TOTAL",
        "Train"     : df["Train"].sum(),
        "Validation": df["Validation"].sum(),
        "Test"      : df["Test"].sum(),
    }])
    df = pd.concat([df, total_row], ignore_index=True)
    print("\nSamples per class per partition:")
    print(df.to_string(index=False))


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(images, labels, idx_train, idx_val, idx_test, train_tf, val_tf, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = CIFAR10Dataset(images[idx_train], labels[idx_train], transform=train_tf)
    val_ds   = CIFAR10Dataset(images[idx_val],   labels[idx_val],   transform=val_tf)
    test_ds  = CIFAR10Dataset(images[idx_test],  labels[idx_test],  transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total = 0
    for imgs, batch_labels in loader:
        imgs, batch_labels = imgs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * imgs.size(0)
        total_correct += out.argmax(1).eq(batch_labels).sum().item()
        total         += imgs.size(0)
    return total_loss / total, total_correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs, batch_labels = imgs.to(device), batch_labels.to(device)
            out           = model(imgs)
            loss          = criterion(out, batch_labels)
            total_loss    += loss.item() * imgs.size(0)
            total_correct += out.argmax(1).eq(batch_labels).sum().item()
            total         += imgs.size(0)
    return total_loss / total, total_correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, patience, save_path, device):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc     = 0.0
    best_epoch       = 0
    patience_counter = 0

    header = (f"{'Epoch':>6} | {'T.Loss':>8} | {'T.Acc':>7} | "
              f"{'V.Loss':>8} | {'V.Acc':>7} | {'LR':>9}")
    print(header)
    print("─" * len(header))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch <= 5 or epoch % 10 == 0:
            print(f"{epoch:>6} | {train_loss:>8.4f} | {100*train_acc:>6.2f}% | "
                  f"{val_loss:>8.4f} | {100*val_acc:>6.2f}% | {lr_now:>9.6f}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc"        : val_acc,
            }, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best val_acc = {100*best_val_acc:.2f}% (epoch {best_epoch})")
            break

    print(f"\nTraining complete. Best val_acc: {100*best_val_acc:.2f}% (epoch {best_epoch})")
    return history, best_val_acc


def plot_curves(history, best_val_acc):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Train vs Validation Curves", fontsize=13, fontweight="bold")

    axes[0].plot(epochs_ran, history["train_loss"], label="Train",      color="steelblue", lw=1.5)
    axes[0].plot(epochs_ran, history["val_loss"],   label="Validation", color="tomato",    lw=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Cross-Entropy)")
    axes[0].set_title("Loss per Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_ran, [100 * a for a in history["train_acc"]], label="Train",      color="steelblue", lw=1.5)
    axes[1].plot(epochs_ran, [100 * a for a in history["val_acc"]],   label="Validation", color="tomato",    lw=1.5)
    axes[1].axhline(y=100 * best_val_acc, color="green", ls="--", lw=1.2,
                    label=f"Best val {100*best_val_acc:.1f}%")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=100, bbox_inches="tight")
    plt.show()


def evaluate_model(model, test_loader, criterion, device, classes, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_acc = {100*checkpoint['val_acc']:.2f}%)")

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest accuracy : {100*test_acc:.2f}%")
    print(f"Test loss     : {test_loss:.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, batch_labels in test_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch_labels.numpy())
    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(all_true, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, ax=ax)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class",      fontsize=12)
    ax.set_title("Confusion Matrix - Test Set", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.show()

    return cm, all_preds, all_true


def discuss_errors(cm, classes):
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    class_acc = cm_norm.diagonal()

    print("Classes with lowest accuracy:")
    for i in np.argsort(class_acc)[:4]:
        print(f"  {classes[i]:12s}: {100*class_acc[i]:.1f}%")

    print("\nMost confused pairs:")
    temp = cm.copy()
    np.fill_diagonal(temp, 0)
    for _ in range(4):
        r, c = np.unravel_index(temp.argmax(), temp.shape)
        print(f"  {classes[r]:12s} -> {classes[c]:12s}: {temp[r, c]} samples")
        temp[r, c] = 0


def main():
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"PyTorch : {torch.__version__}")

    print("\n--- 1. Dataset Loading ---")
    images, labels = load_dataset()
    show_dataset_info(images, labels, CLASSES)
    show_samples(images, labels, CLASSES, RANDOM_STATE)

    print("\n--- 2. Preprocessing & Data Augmentation ---")
    train_tf, val_tf = get_transforms()

    print("\n--- 3. Data Split ---")
    idx_train, idx_val, idx_test = split_data(images, labels, RANDOM_STATE)
    show_split_table(labels, idx_train, idx_val, idx_test, CLASSES)
    train_loader, val_loader, test_loader = get_dataloaders(
        images, labels, idx_train, idx_val, idx_test, train_tf, val_tf, RANDOM_STATE
    )

    print("\n--- 4. CNN Architecture ---")
    model = CIFAR10CNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(model)

    print("\n--- 5. Training ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=200, patience=10, save_path="best_model.pth", device=device
    )
    plot_curves(history, best_val_acc)

    print("\n--- 6. Test Evaluation ---")
    cm, all_preds, all_true = evaluate_model(
        model, test_loader, criterion, device, CLASSES, "best_model.pth"
    )
    discuss_errors(cm, CLASSES)


if __name__ == "__main__":
    main()
