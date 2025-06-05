import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
import json
from train_baseline import ResNetClassifier

def evaluate_model(data_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test data
    data = torch.load(data_path)
    x, y = data['features'], data['labels']
    unique, counts = torch.unique(y, return_counts=True)
    print("Label counts:", dict(zip(unique.tolist(), counts.tolist())))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)

    # model
    model = ResNetClassifier(input_dim=x.shape[1], num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds).tolist()

    print(f"Test Accuracy: {acc:.4f}")
    with open(output_path, 'w') as f:
        json.dump({'accuracy': acc, 'confusion_matrix': cm}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/test.pt')
    parser.add_argument('--model', type=str,  default='models/dropout_stratified.pth')
    parser.add_argument('--output', type=str, default='results/test_metrics.json')
    args = parser.parse_args()

    evaluate_model(args.data, args.model, args.output)
