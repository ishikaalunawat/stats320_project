import torch
import matplotlib.pyplot as plt

label_names = ['other', 'investigation', 'mount', 'attack']

def plot_label_distribution(path, title="Label Distribution"):
    data = torch.load(path)
    labels = data['labels'].numpy()

    counts = [sum(labels == i) for i in range(len(label_names))]

    plt.bar(label_names, counts)
    plt.title(title)
    plt.ylabel("Count")
    plt.savefig(f'results/{title}_label_distribution.png')

if __name__ == "__main__":
    plot_label_distribution("data/processed/data.pt", "Train Set")
    plot_label_distribution("data/processed/test.pt", "Test Set")
