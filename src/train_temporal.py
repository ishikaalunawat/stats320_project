import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import os
import argparse
import numpy as np
import torch.backends.cudnn
torch.backends.cudnn.enabled = False

class TemporalCNN(nn.Module):
    def __init__(self, in_dim=316, hidden_dim=128, time_dim=11, num_classes=4):
        super().__init__()

        self.temporal_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),  # [B, 128, 11]
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),          # [B, 64, 11]
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                               # [B, 64, 1]
            nn.Flatten(),                                          # [B, 64]
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, 316, 11] → [B, 11, 316]
        x = x.permute(0, 2, 1)
        x = self.temporal_proj(x)         # [B, 11, 128]
        x = x.permute(0, 2, 1)            # [B, 128, 11]
        return self.net(x)


def get_class_weights(labels, num_classes):
    weights = compute_class_weight(class_weight='balanced',
                                    classes=np.array(range(num_classes)),
                                    y=labels.numpy())
    return torch.tensor(weights, dtype=torch.float32)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(yb.numpy())
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds).tolist()
    return acc, cm

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = torch.load(args.data)
    x, y = data['features'], data['labels']
    dataset = TensorDataset(x, y)

    # Train/val split
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = TemporalCNN().to(device)

    # Loss with class weights
    class_weights = get_class_weights(y, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0
    patience = 5
    counter = 0

    for epoch in range(30):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc, cm = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break
    # final eval
    acc, cm = evaluate(model, train_loader, device)
    print("Final accuracy:", acc)
    with open('results/train_metrics_temporal.json', 'w') as f:
        json.dump({'accuracy': acc, 'confusion_matrix': cm}, f)

    # Final save
    print("Best Validation Accuracy:", best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/data_seq.pt')
    parser.add_argument('--model', type=str, default='models/temporal_cnn.pth')
    args = parser.parse_args()
    main(args)
