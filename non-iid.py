"""
fedavg_pneumonia_resnet_non_iid.py

Non-IID FedAvg ResNet script:
- Dirichlet non-IID split across clients
- FedAvg aggregation
- Outputs saved to ./non_iid_results/
    - global_accuracy_vs_rounds.png
    - roc_curve_global.png
    - client_accuracy_over_rounds.png
    - final_accuracy_per_client_bar.png
    - per_client_final_results.csv
    - final_global_metrics.txt
"""

import os
import copy
import random
from pathlib import Path
from tqdm import trange, tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

# -----------------------
# Config / Hyperparams
# -----------------------
DATA_DIR = "./data/chest_xray"   # path to chest_xray folder containing train/ test/ (val optional)
RESULTS_DIR = "./non_iid_results"  # where outputs will be saved
NUM_CLIENTS = 4                    # number of simulated hospitals/clients
NUM_ROUNDS = 20                    # communication rounds
FRACTION_CLIENTS = 1.0             # fraction of clients sampled per round (1.0 -> full participation)
LOCAL_EPOCHS = 2                   # local epochs per client
LOCAL_BATCH_SIZE = 8               # keep small for CPU
LR = 5e-4                          # learning rate
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0                    # set to 0 for Windows/CPU to avoid multiprocessing spawn issues
SEED = 42
DIRICHLET_ALPHA = 0.5              # alpha <1 -> more heterogeneous (non-iid). Increase toward uniform.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create results dir
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------
# Utility functions and model definition
# -----------------------

def create_model(pretrained=True, freeze_backbone=True):
    """
    Create ResNet18 using modern torchvision API if available (avoids deprecation warnings).
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)
    if freeze_backbone:
        # freeze all except fc and batchnorm parameters (BN should keep running stats)
        for name, p in model.named_parameters():
            if "fc" not in name and "bn" not in name:
                p.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def local_train(model, dataloader, device, epochs=1, lr=1e-3, weight_decay=WEIGHT_DECAY):
    """
    Train local model for a few epochs and return state_dict.
    Uses Adam optimizer; only parameters with requires_grad=True are optimized.
    """
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
            # optional: clip grads to be safe
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model.state_dict()

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate model on provided dataloader. Returns accuracy, f1, auc, and raw predictions.
    """
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
        auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, zero_division=0)
    return {"accuracy": acc, "f1": f1, "auc": auc, "y_true": ys, "y_prob": probs}

def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Global Model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def non_iid_dirichlet_split(dataset, num_clients, alpha=0.5, seed=42):
    """
    Create a non-iid split of indices for dataset according to Dirichlet distribution per class.
    Returns: list of index lists, one per client.
    """
    np.random.seed(seed)
    # get labels
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]
    # For each class, split indices using Dirichlet proportions
    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        np.random.shuffle(cls_idx)
        # draw proportions for each client from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        # scale proportions by number of samples and round to ints
        # ensure each client gets at least one sample if possible
        counts = (proportions * len(cls_idx)).astype(int)
        # fix rounding errors: distribute remaining samples
        remainder = len(cls_idx) - counts.sum()
        if remainder > 0:
            for i in np.argsort(-proportions)[:remainder]:
                counts[i] += 1
        # assign slices
        start = 0
        for i in range(num_clients):
            cnt = counts[i]
            if cnt > 0:
                client_indices[i].extend(cls_idx[start:start+cnt].tolist())
            start += cnt
    # final check: some clients may be empty (rare); if empty distribute random samples
    for i in range(num_clients):
        if len(client_indices[i]) == 0:
            # give one random sample from dataset
            idx = np.random.choice(len(dataset))
            client_indices[i].append(int(idx))
    return client_indices

def get_loader_from_indices(dataset, indices, batch_size, shuffle_):
    """
    Shallow copy ImageFolder dataset and set samples/targets according to indices.
    Return a DataLoader.
    """
    ds = copy.copy(dataset)
    ds.samples = [dataset.samples[i] for i in indices]
    ds.targets = [dataset.targets[i] for i in indices]
    ds.transform = dataset.transform  # keep transform already set on the dataset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_, num_workers=NUM_WORKERS, pin_memory=False)

# -----------------------
# Main
# -----------------------
def main():
    print("Device:", DEVICE)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # transforms
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

    # load datasets
    train_root = os.path.join(DATA_DIR, "train")
    test_root = os.path.join(DATA_DIR, "test")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train folder not found at {train_root}")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test folder not found at {test_root}")

    full_train = datasets.ImageFolder(train_root, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_root, transform=eval_transform)
    print("Classes (train):", full_train.class_to_idx)

    # make non-iid split
    client_splits = non_iid_dirichlet_split(full_train, NUM_CLIENTS, alpha=DIRICHLET_ALPHA, seed=SEED)
    print("Non-IID Dirichlet split (alpha=%.3f) sizes:" % DIRICHLET_ALPHA, [len(s) for s in client_splits])

    # split each client's indices into local train/val (80/20)
    clients = []
    for s in client_splits:
        random.shuffle(s)
        cutoff = int(0.8 * len(s))
        train_idxs = s[:cutoff]
        val_idxs = s[cutoff:]
        clients.append({"train_idxs": train_idxs, "val_idxs": val_idxs})

    print("Client dataset sizes (train,val):", [(len(c['train_idxs']), len(c['val_idxs'])) for c in clients])

    # test loader (shared)
    test_loader = DataLoader(test_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # build global model (pretrained)
    global_model = create_model(pretrained=True, freeze_backbone=True).to(DEVICE)
    global_weights = copy.deepcopy(global_model.state_dict())

    # histories
    global_acc_history = []
    global_f1_history = []
    global_auc_history = []
    client_test_acc_history = {i: [] for i in range(NUM_CLIENTS)}

    # main federated loop
    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n=== Communication Round {rnd}/{NUM_ROUNDS} ===")
        # select clients
        if FRACTION_CLIENTS >= 1.0:
            selected_clients = list(range(NUM_CLIENTS))
        else:
            m = max(1, int(FRACTION_CLIENTS * NUM_CLIENTS))
            selected_clients = sorted(random.sample(range(NUM_CLIENTS), m))
        print("Selected clients:", selected_clients)

        # init placeholders for this round
        for cid in range(NUM_CLIENTS):
            client_test_acc_history[cid].append(np.nan)

        local_weights = []
        local_sizes = []

        for c in selected_clients:
            train_idxs = clients[c]["train_idxs"]
            if len(train_idxs) == 0:
                print(f"Client {c} empty train, skipping")
                continue

            loader = get_loader_from_indices(full_train, train_idxs, LOCAL_BATCH_SIZE, shuffle_=True)

            # local model init with global weights
            local_model = create_model(pretrained=False, freeze_backbone=True).to(DEVICE)
            local_model.load_state_dict(global_weights)

            # train locally
            updated_weights = local_train(local_model, loader, DEVICE, epochs=LOCAL_EPOCHS, lr=LR)
            local_weights.append({k: v.cpu().clone() for k, v in updated_weights.items()})
            local_sizes.append(len(train_idxs))

            # evaluate local updated model on shared test set (diagnostic)
            client_metrics = evaluate_model(local_model, test_loader, DEVICE)
            client_test_acc_history[c][-1] = client_metrics["accuracy"]
            print(f"  Client {c} updated -> test acc: {client_metrics['accuracy']:.4f}")

        if len(local_weights) == 0:
            print("No local updates this round, skipping aggregation.")
            continue

        # FedAvg aggregation (weighted)
        total_samples = sum(local_sizes)
        new_global = {}
        first = local_weights[0]
        for k in first.keys():
            if first[k].dtype == torch.float32:
                new_global[k] = torch.zeros_like(first[k])
            else:
                new_global[k] = first[k].clone()
        for lw, size in zip(local_weights, local_sizes):
            w = size / total_samples
            for k in lw.keys():
                if lw[k].dtype == torch.float32:
                    new_global[k] += lw[k] * w

        # set global weights
        global_weights = {k: new_global[k].to(DEVICE) for k in new_global.keys()}
        global_model.load_state_dict(global_weights)

        # evaluate global on shared test set
        metrics = evaluate_model(global_model, test_loader, DEVICE)
        print(f"Global aggregated -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        global_acc_history.append(metrics['accuracy'])
        global_f1_history.append(metrics['f1'])
        global_auc_history.append(metrics['auc'])

    # final global metrics
    if len(global_acc_history) > 0:
        final_metrics = {"accuracy": global_acc_history[-1], "f1": global_f1_history[-1], "auc": global_auc_history[-1]}
    else:
        final_metrics = {"accuracy": float("nan"), "f1": float("nan"), "auc": float("nan")}

    print("\n=== Final Global Model Metrics ===")
    print(final_metrics)
    with open(os.path.join(RESULTS_DIR, "final_global_metrics.txt"), "w") as f:
        f.write(str(final_metrics))

    # ROC for final global model
    test_eval = evaluate_model(global_model, test_loader, DEVICE)
    if len(np.unique(test_eval["y_true"])) > 1:
        plot_roc_curve(test_eval["y_true"], test_eval["y_prob"], save_path=os.path.join(RESULTS_DIR, "roc_curve_global.png"))
        print("Saved ROC curve")
    else:
        print("ROC not plotted (single-class in test set)")

    # Per-client final summary (compute mean/var over rounds where they participated)
    per_client_results = []
    for cid in range(NUM_CLIENTS):
        accs = np.array([x for x in client_test_acc_history[cid] if not np.isnan(x)])
        if accs.size > 0:
            final_acc = float(accs[-1])
            mean_acc = float(np.mean(accs))
            var_acc = float(np.var(accs))
            std_acc = float(np.std(accs))
            participated = int(accs.size)
        else:
            final_acc = mean_acc = var_acc = std_acc = float("nan")
            participated = 0
        per_client_results.append({
            "client": cid,
            "final_test_accuracy": final_acc,
            "mean_test_accuracy": mean_acc,
            "variance_test_accuracy": var_acc,
            "std_test_accuracy": std_acc,
            "num_rounds_participated": participated,
            "n_local_train_samples": len(clients[cid]["train_idxs"])
        })
        print(f"Client {cid}: final_acc={final_acc:.4f}, mean={mean_acc:.4f}, var={var_acc:.6f}, rounds={participated}")

    # save per-client CSV
    results_df = pd.DataFrame(per_client_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "per_client_final_results.csv"), index=False)
    print("Saved per_client_final_results.csv")

    # Plot global accuracy vs rounds
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(global_acc_history)+1), global_acc_history, marker='o', linewidth=2, label='Global Accuracy')
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Global Model Accuracy vs Communication Rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "global_accuracy_vs_rounds.png"), dpi=150)
    plt.close()
    print("Saved global_accuracy_vs_rounds.png")

    # Plot client-wise accuracy trajectories (smoothed)
    plt.figure(figsize=(12,7))
    colors = plt.cm.Set2(np.linspace(0,1,NUM_CLIENTS)) if NUM_CLIENTS <= 8 else plt.cm.tab10(np.linspace(0,1,NUM_CLIENTS))
    for cid in range(NUM_CLIENTS):
        rounds = list(range(1, len(client_test_acc_history[cid])+1))
        accs = client_test_acc_history[cid]
        # EMA smoothing
        smoothed = []
        ema = None
        alpha = 0.35
        for a in accs:
            if np.isnan(a):
                smoothed.append(np.nan)
            else:
                if ema is None:
                    ema = a
                else:
                    ema = alpha * a + (1-alpha) * ema
                smoothed.append(ema)
        plt.plot(rounds, smoothed, marker='o', linewidth=2, markersize=6, alpha=0.85, color=colors[cid], label=f"Client {cid}")

    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (shared test set)")
    plt.title("Client-wise Accuracy over Federated Rounds (EMA smoothed)")
    plt.legend(loc='best', ncol=min(3, NUM_CLIENTS))
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, NUM_ROUNDS+1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "client_accuracy_over_rounds.png"), dpi=150)
    plt.close()
    print("Saved client_accuracy_over_rounds.png")

    # Final Test Accuracy per Client (bar graph)
    client_df = results_df.copy()
    plt.figure(figsize=(8,5))
    plt.bar(client_df["client"].astype(str), client_df["final_test_accuracy"], color=plt.cm.Set2(np.linspace(0,1,NUM_CLIENTS)))
    plt.xlabel("Client ID")
    plt.ylabel("Final Test Accuracy (shared test set)")
    plt.title("Final Test Accuracy per Client")
    plt.ylim(0,1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_accuracy_per_client_bar.png"), dpi=150)
    plt.close()
    print("Saved final_accuracy_per_client_bar.png")

    print("\nAll results saved to:", RESULTS_DIR)
    print("Files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(" -", f)

if __name__ == "__main__":
    main()
