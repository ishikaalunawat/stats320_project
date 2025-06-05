import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(labels, num_classes):
    weights = compute_class_weight(class_weight='balanced',
                                    classes=np.arange(num_classes),
                                    y=labels.numpy())
    return torch.tensor(weights, dtype=torch.float32)

# ResNet block for feature vectors
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))

# baseline ResNet-style model with features input
class ResNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.res2 = ResidualBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        x = self.dropout(x)
        return self.output_layer(x)

# train
def train(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        noise = torch.normal(0, 0.02, size=inputs.shape).to(device)
        inputs = inputs + noise
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    writer.add_scalar('Train/Loss', running_loss / len(dataloader), epoch)

# eval
def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.numpy())
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds).tolist()
    return acc, cm

# main
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/data.pt')
    parser.add_argument('--output', type=str, default='models/baseline_stratified.pth')
    args = parser.parse_args()

    # Load data
    data = torch.load(args.data)
    features, labels = data['features'], data['labels']
    print(f"Loaded data with {features.shape[0]} samples and {features.shape[1]} features")
    unique, counts = torch.unique(labels, return_counts=True)
    print("Label counts:", dict(zip(unique.tolist(), counts.tolist())))
    # Split into train/val
    train_X, val_X, train_y, val_y = train_test_split(
        features, labels, test_size=0.1, stratify=labels, random_state=42
    )


    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=64)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetClassifier(input_dim=features.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    weights = get_class_weights(train_y, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    writer = SummaryWriter(log_dir='runs/baseline_stratified')

    best_val_acc = 0
    patience, patience_counter = 5, 0

    for epoch in range(50):
        print(f"Epoch {epoch+1}")
        train(model, train_loader, optimizer, criterion, device, writer, epoch)

        # Evaluate on validation set
        val_acc, val_cm = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")
        writer.add_scalar('Val/Accuracy', val_acc, epoch)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), args.output)
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print("Early stopping")
        #         break

    # Final metrics
    acc, cm = evaluate(model, val_loader, device)
    print("Final validation accuracy:", acc)
    with open('results/train_metrics.json', 'w') as f:
        json.dump({'val_accuracy': acc, 'val_confusion_matrix': cm}, f)

if __name__ == "__main__":
    main()
