import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
import json
from train_baseline import ResNetClassifier

def evaluate_model(x, y, model, device):
    '''
    Evaluate the model given inputs and outputs
    '''
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)

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

    return acc

def feature_importance(data_path, model_path, metric_path, output_path, 
                       n_shuffles=10):
    '''
    Given feature matrix, shuffle each feature n_shuffles times and measure the
    drop in model accuracy. Take the average across shuffles and compare the
    results for each feature.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Accuracy before shuffling features
    with open(metric_path, 'r') as f:
        initial_accuracy = json.load(f)['accuracy']


    # Test Data
    data = torch.load(data_path)
    x, y = data['features'], data['labels']


    # Model
    model = ResNetClassifier(input_dim=x.shape[1], num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()


    # Shuffle
    n_samples, n_features = x.shape

    # Tensor to house mean drop for each feature
    mean_deltas = torch.zeros(n_features)
    
    for i in range(n_features):
        # Tensor to house drop for each shuffle
        deltas = torch.zeros(n_shuffles)

        for j in range(n_shuffles):
            perm = torch.randperm(n_samples)
            new_data = x.detach().clone()
            new_data[:, i] = new_data[perm, i]

            deltas[j] = initial_accuracy - evaluate_model(x, y, model, device)

        mean_deltas[i] = deltas.mean()

    sorted, rankings = mean_deltas.sort(descending=True)[1]
        
    with open(output_path, 'w') as f:
        json.dump({'deltas': sorted, 'rankings': rankings}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/test.pt')
    parser.add_argument('--model', type=str,  default='models/baseline_stratified.pth')
    parser.add_argument('--metric', type=str, default='results/baseline_stratified/test_metrics.json')
    parser.add_argument('--output', type=str, default='results/baseline_stratified/feature_importance.json')
    args = parser.parse_args()

    evaluate_model(args.data, args.model, args.metric, args.output)
