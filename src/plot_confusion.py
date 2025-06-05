import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

label_names = ['other', 'investigation', 'mount', 'attack']

def plot_confusion_matrix(json_path, title='CF', normalize=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if title=='train':
        cm = np.array(data['train_confusion_matrix'])
        acc = data['train_accuracy']
    else:
        cm = np.array(data['confusion_matrix'])
        acc = data['accuracy']
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({title} acc.: {acc:.2%})")
    plt.tight_layout()
    plt.savefig(f"{'/'.join(json_path.split('/')[:2])}/confusion_matrix_{title}.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline_stratified', help='Model type')
    args = parser.parse_args()
    plot_confusion_matrix(f"results/{args.model}/test_metrics.json", 'test', normalize=True)
    plot_confusion_matrix(f"results/{args.model}/train_metrics.json", 'train', normalize=True)
