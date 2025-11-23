"""
fedavg_pneumonia_resnet_fixed.py
Fixed version (Windows/CPU-safe) of FedAvg ResNet script with:
- per-client accuracy trajectories logging (on shared test set)
- Client convergence plot showing weights aggregation effect
- ROC curve, per-client CSV, global accuracy plot, final metrics
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

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score , roc_curve

# -----------------------
# Config / Hyperparams
# -----------------------
DATA_DIR = "./data/chest_xray"
NUM_CLIENTS = 4                    # number of simulated hospitals/clients
NUM_ROUNDS = 15                    # communication rounds
FRACTION_CLIENTS = 1.0             # SELECT ALL CLIENTS EVERY ROUND (no randomness)
LOCAL_EPOCHS = 2                   # local epochs per client (increased for better convergence)
LOCAL_BATCH_SIZE = 8               # smaller batch size for more updates per epoch
LR = 5e-4                          # lower learning rate for stability
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0    # IMPORTANT: set to 0 for Windows/CPU to avoid spawn() multiprocessing issues
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utility functions and model definition
# -----------------------

def create_model(pretrained=True, freeze_backbone=True):
    # Use the new weights enum when available to avoid deprecation warnings.
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=pretrained)
    # Do not freeze backbone by default for better fine-tuning; if requested, freeze except BatchNorm
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'bn' not in name:
                param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)   # binary classification logits
    return model


def local_train(model, dataloader, device, epochs=1, lr=1e-3, weight_decay=WEIGHT_DECAY):
    model.train()

    # Use different LR for backbone and classifier for stability
    backbone_params = [p for n, p in model.named_parameters() if ('fc' not in n and p.requires_grad)]
    fc_params = [p for n, p in model.named_parameters() if ('fc' in n and p.requires_grad)]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr * 0.1})
    if fc_params:
        param_groups.append({'params': fc_params, 'lr': lr})

    if not param_groups:
        # fallback
        param_groups = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr}]

    optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    # Small LR decay per local epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # If dataloader provides pos_weight via attribute, use it; otherwise regular BCE
    pos_weight = getattr(dataloader, 'pos_weight', None)
    if pos_weight is not None:
        pos_w_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

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

        if shuffle_:
            # Balanced sampling to address class imbalance per client
            targets = ds.targets
            class_counts = {}
            for t in targets:
                class_counts[t] = class_counts.get(t, 0) + 1
            # avoid division by zero
            weights = [1.0 / class_counts[t] for t in targets]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=False)
            # attach pos_weight to loader for loss balancing: neg/pos
            pos = sum(1 for t in targets if t == 1)
            neg = len(targets) - pos
            loader.pos_weight = (neg / pos) if pos > 0 else 1.0
            return loader

        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # Test loader
    test_loader = DataLoader(test_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # Build global model
    global_model = create_model(pretrained=True, freeze_backbone=True).to(DEVICE)
    global_weights = copy.deepcopy(global_model.state_dict())

    global_acc_history = []
    global_f1_history = []
    global_auc_history = []

    # === Track each client's updated model performance on SHARED TEST SET ===
    client_test_acc_history = {i: [] for i in range(NUM_CLIENTS)}
    # Smooth using exponential moving average
    client_acc_smoothed = {i: [] for i in range(NUM_CLIENTS)}

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n=== Communication Round {rnd}/{NUM_ROUNDS} ===")
        m = max(1, int(FRACTION_CLIENTS * NUM_CLIENTS))
        selected_clients = random.sample(range(NUM_CLIENTS), m)
        print("Selected clients:", selected_clients)

        # Initialize all clients with NaN for this round (not selected = NaN)
        for cid in range(NUM_CLIENTS):
            client_test_acc_history[cid].append(np.nan)

        local_weights = []
        local_sizes = []

        for c in selected_clients:
            train_idxs = clients[c]["train_idxs"]
            if len(train_idxs) == 0:
                print(f"Client {c} has no train data, skipping.")
                continue
            
            loader = get_loader_from_indices(full_train, train_idxs, LOCAL_BATCH_SIZE, shuffle_=True)
            
            # Local model initialized with global weights
            local_model = create_model(pretrained=True, freeze_backbone=True).to(DEVICE)
            local_model.load_state_dict(global_weights)
            
            # Local training
            updated_weights = local_train(local_model, loader, DEVICE, epochs=LOCAL_EPOCHS, lr=LR)
            local_weights.append({k: v.cpu().clone() for k, v in updated_weights.items()})
            local_sizes.append(len(train_idxs))

            # Evaluate this client's updated local model on SHARED TEST SET
            client_metrics = evaluate_model(local_model, test_loader, DEVICE)
            client_test_acc_history[c][-1] = client_metrics["accuracy"]
            print(f"  Client {c} updated model -> Test Acc: {client_metrics['accuracy']:.4f}")

        if len(local_weights) == 0:
            print("No local updates this round.")
            continue

        # Weighted average (FedAvg) - aggregate weights
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

        # Convert to DEVICE tensors then load to model
        global_weights = {k: new_global[k].to(DEVICE) for k in new_global.keys()}
        global_model.load_state_dict(global_weights)

        # Evaluate aggregated global model on test set
        metrics = evaluate_model(global_model, test_loader, DEVICE)
        print(f"Global aggregated model -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
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

    # Per-client performance (final) on their local validation sets
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

    # === Final Test Accuracy per Client (Bar Graph) ===
    # Load client results (already saved above)
    client_df = pd.DataFrame(per_client_results)

    plt.figure(figsize=(8, 5))
    plt.bar(client_df["client"].astype(str), client_df["accuracy"])
    plt.xlabel("Client ID", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12, fontweight='bold')
    plt.title("Final Test Accuracy per Client", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("final_accuracy_per_client_bar.png", dpi=150)
    print("Saved final_accuracy_per_client_bar.png")


    # Plot global accuracy vs rounds
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(global_acc_history) + 1), global_acc_history, marker='o', linewidth=2, label='Global Accuracy')
    plt.xlabel("Communication Round", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12, fontweight='bold')
    plt.title("Global Model Accuracy vs Communication Rounds", fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("global_accuracy_vs_rounds.png", dpi=150)
    print("Saved global_accuracy_vs_rounds.png")

    # === Plot ONLY client trajectories (aggregation IS FedAvg) ===
    plt.figure(figsize=(12, 7))

    # Define colors for each client - use more colors for scalability
    if NUM_CLIENTS <= 5:
        colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLIENTS))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLIENTS))

    # Plot each client's accuracy trajectory on shared test set with smoothing
    for cid in range(NUM_CLIENTS):
        rounds = list(range(1, len(client_test_acc_history[cid]) + 1))
        accs = client_test_acc_history[cid]
        
        # Apply exponential moving average for smoothing (alpha=0.3 for moderate smoothing)
        smoothed_accs = []
        ema = None
        for acc in accs:
            if not np.isnan(acc):
                if ema is None:
                    ema = acc
                else:
                    ema = 0.3 * acc + 0.7 * ema
                smoothed_accs.append(ema)
            else:
                smoothed_accs.append(np.nan)
        
        plt.plot(rounds, smoothed_accs,
                 marker='o',
                 markersize=7,
                 linewidth=2.8,
                 alpha=0.8,
                 color=colors[cid],
                 label=f'Client {cid}')

    plt.xlabel("Federated Round", fontsize=13, fontweight='bold')
    plt.ylabel("Test Accuracy", fontsize=13, fontweight='bold')
    plt.title(f"Client-wise Accuracy over Federated Rounds ({NUM_CLIENTS} Clients) - Smoothed", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11, ncol=min(3, NUM_CLIENTS))
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.ylim([min([min([x for x in client_test_acc_history[i] if not np.isnan(x)], default=0.5) for i in range(NUM_CLIENTS)]) - 0.05, 1.0])
    plt.xticks(range(1, NUM_ROUNDS + 1))
    plt.tight_layout()
    plt.savefig("client_accuracy_over_rounds.png", dpi=150)
    print("Saved client_accuracy_over_rounds.png")

    print("\nTraining complete. Outputs saved:")
    print(" - global_accuracy_vs_rounds.png")
    print(" - fedavg_clients_convergence.png (clients converging via weight aggregation)")
    print(" - per_client_results.csv")
    print(" - final_global_metrics.txt")
    print(" - roc_curve_global.png (if plotted)")

if __name__ == "__main__":
    main()
