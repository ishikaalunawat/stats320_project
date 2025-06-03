import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

label_names = ['other', 'investigation', 'mount', 'attack']

def plot_confusion_matrix(json_path, title='CF', normalize=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    cm = np.array(data['confusion_matrix'])
    acc = data['accuracy']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.tight_layout()
    plt.savefig(f'results/{title}_confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    plot_confusion_matrix("results/test_metrics.json", 'Test', normalize=True)
    plot_confusion_matrix("results/baseline_metrics.json", 'Train', normalize=True)
