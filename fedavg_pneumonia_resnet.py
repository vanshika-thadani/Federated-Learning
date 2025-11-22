"""
fedavg_pneumonia_resnet_fixed.py
Fixed version (Windows/CPU-safe) of FedAvg ResNet script.
- Wraps main training loop inside main() and protects with if __name__ == '__main__'
- Sets NUM_WORKERS = 0 and pin_memory=False to avoid multiprocessing spawn issues on Windows
- Keeps pretrained ResNet18 backbone frozen by default
- Saves outputs: global_accuracy_vs_rounds.png, per_client_results.csv, final_global_metrics.txt
"""

import os
import copy
import random
from tqdm import trange, tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -----------------------
# Config / Hyperparams
# -----------------------
DATA_DIR = "./data/chest_xray"   # change if needed
NUM_CLIENTS = 5                    # number of simulated hospitals/clients
NUM_ROUNDS = 10                    # communication rounds
FRACTION_CLIENTS = 0.6             # fraction of clients sampled per round
LOCAL_EPOCHS = 2                   # local epochs per client
LOCAL_BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
IMAGE_SIZE = 224
NUM_WORKERS = 0    # IMPORTANT: set to 0 for Windows/CPU to avoid spawn() multiprocessing issues
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utility functions and model definition
# -----------------------

def create_model(pretrained=True, freeze_backbone=True):
    model = models.resnet18(pretrained=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)   # binary classification logits
    return model


def local_train(model, dataloader, device, epochs=1, lr=1e-3, weight_decay=WEIGHT_DECAY):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    ys = []
    probs = []
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        ps = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
        probs.extend(ps.tolist())
        ys.extend(labels.numpy().tolist())
    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float('nan')
    except Exception:
        auc = float('nan')
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, zero_division=0)
    return {"accuracy": acc, "f1": f1, "auc": auc, "y_true": ys, "y_prob": probs}

def plot_roc_curve(y_true, y_prob, save_path="roc_curve_global.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0,1], [0,1], linestyle="--")  # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Global Model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")


# -----------------------
# Main function (protected)
# -----------------------
def main():
    print("Using device:", DEVICE)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Load datasets
    train_root = os.path.join(DATA_DIR, "train")
    test_root = os.path.join(DATA_DIR, "test")

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train folder not found at {train_root}")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test folder not found at {test_root}")

    full_train = datasets.ImageFolder(train_root, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_root, transform=eval_transform)
    print("Classes (train):", full_train.class_to_idx)

    # IID split across clients
    def iid_split(dataset, num_clients):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return [list(s) for s in splits]

    client_splits = iid_split(full_train, NUM_CLIENTS)

    # For each client, split into local train/val (80/20)
    clients = []
    for s in client_splits:
        random.shuffle(s)
        cutoff = int(0.8 * len(s))
        train_idxs = s[:cutoff]
        val_idxs = s[cutoff:]
        clients.append({"train_idxs": train_idxs, "val_idxs": val_idxs})

    print("Client dataset sizes (train,val):", [(len(c['train_idxs']), len(c['val_idxs'])) for c in clients])

    # Helper to create a DataLoader for subset indices
    def get_loader_from_indices(dataset, indices, batch_size, shuffle_):
        ds = copy.copy(dataset)  # shallow copy
        ds.samples = [dataset.samples[i] for i in indices]
        ds.targets = [dataset.targets[i] for i in indices]
        ds.transform = train_transform if shuffle_ else eval_transform
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_, num_workers=NUM_WORKERS, pin_memory=False)

    # Test loader
    test_loader = DataLoader(test_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # Build global model
    global_model = create_model(pretrained=True, freeze_backbone=True).to(DEVICE)
    global_weights = copy.deepcopy(global_model.state_dict())

    global_acc_history = []
    global_f1_history = []
    global_auc_history = []

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n=== Communication Round {rnd}/{NUM_ROUNDS} ===")
        m = max(1, int(FRACTION_CLIENTS * NUM_CLIENTS))
        selected_clients = random.sample(range(NUM_CLIENTS), m)
        print("Selected clients:", selected_clients)

        local_weights = []
        local_sizes = []

        for c in selected_clients:
            train_idxs = clients[c]["train_idxs"]
            if len(train_idxs) == 0:
                print(f"Client {c} has no train data, skipping.")
                continue
            loader = get_loader_from_indices(full_train, train_idxs, LOCAL_BATCH_SIZE, shuffle_=True)
            # local model initialized with global weights
            local_model = create_model(pretrained=True, freeze_backbone=True).to(DEVICE)
            local_model.load_state_dict(global_weights)
            # local training
            updated_weights = local_train(local_model, loader, DEVICE, epochs=LOCAL_EPOCHS, lr=LR)
            local_weights.append({k: v.cpu().clone() for k, v in updated_weights.items()})
            local_sizes.append(len(train_idxs))

        if len(local_weights) == 0:
            print("No local updates this round.")
            continue

        # weighted average (FedAvg)
        total_samples = sum(local_sizes)
        new_global = {}
        for key in local_weights[0].keys():
        # Only average float tensors (weights), skip integer buffers
            if local_weights[0][key].dtype == torch.float32:
                new_global[key] = torch.zeros_like(local_weights[0][key])
            else:
                # copy first client's value (they are identical across clients)
                new_global[key] = local_weights[0][key].clone()

        for lw, size in zip(local_weights, local_sizes):
            for k in lw.keys():
                if lw[k].dtype == torch.float32:
                    new_global[k] += lw[k] * (size / total_samples)


        # convert to DEVICE tensors then load to model
        global_weights = {k: new_global[k].to(DEVICE) for k in new_global.keys()}
        global_model.load_state_dict(global_weights)

        # evaluate on global test set
        metrics = evaluate_model(global_model, test_loader, DEVICE)
        print(f"Global Test -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        global_acc_history.append(metrics['accuracy'])
        global_f1_history.append(metrics['f1'])
        global_auc_history.append(metrics['auc'])

    # Final metrics on global test
    if len(global_acc_history) > 0:
        final_metrics = {"accuracy": global_acc_history[-1], "f1": global_f1_history[-1], "auc": global_auc_history[-1]}
    else:
        final_metrics = {"accuracy": float('nan'), "f1": float('nan'), "auc": float('nan')}

    print("\n=== Final Global Model Metrics ===")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"AUC: {final_metrics['auc']:.4f}")
    with open("final_global_metrics.txt", "w") as f:
        f.write(str(final_metrics))

    # === Generate ROC Curve for Final Global Model ===
    test_eval = evaluate_model(global_model, test_loader, DEVICE)

    if len(np.unique(test_eval["y_true"])) > 1:
        plot_roc_curve(test_eval["y_true"], test_eval["y_prob"])
    else:
        print("ROC Curve cannot be plotted: Test labels contain only one class.")


    # Per-client performance
    per_client_results = []
    for i, c in enumerate(clients):
        val_idxs = c["val_idxs"]
        if len(val_idxs) == 0:
            per_client_results.append({"client": i, "n_samples": 0, "accuracy": float('nan'), "f1": float('nan'), "auc": float('nan')})
            continue
        val_loader = get_loader_from_indices(full_train, val_idxs, LOCAL_BATCH_SIZE, shuffle_=False)
        res = evaluate_model(global_model, val_loader, DEVICE)
        per_client_results.append({"client": i, "n_samples": len(val_idxs), "accuracy": res["accuracy"], "f1": res["f1"], "auc": res["auc"]})
        print(f"Client {i} (n={len(val_idxs)}) -> Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, AUC: {res['auc']:.4f}")

    pd.DataFrame(per_client_results).to_csv("per_client_results.csv", index=False)
    print("Saved per_client_results.csv")

    

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(global_acc_history)+1), global_acc_history, marker='o', label='Global Accuracy')
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Global Model Accuracy vs Communication Rounds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("global_accuracy_vs_rounds.png")
    print("Saved global_accuracy_vs_rounds.png")

    print("Training complete. Outputs saved:")
    print(" - global_accuracy_vs_rounds.png")
    print(" - per_client_results.csv")
    print(" - final_global_metrics.txt")


if __name__ == "__main__":
    main()
