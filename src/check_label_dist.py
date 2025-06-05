import torch
import matplotlib.pyplot as plt

# last three labels from data_reader.py and 0 is 'other'
ind2label = {0: 'other', 1: 'investigation', 2: 'mount', 3: 'attack'}
label_names = ['other', 'investigation', 'mount', 'attack']

def plot_label_distribution(path, title="Label Distribution"):
    data = torch.load(path)
    labels = data['labels'].numpy()
    counts = [sum(labels == i) for i in range(len(label_names))]
    
    plt.figure()
    plt.bar(label_names, counts)
    plt.title(title)
    plt.ylabel("Count")
    plt.savefig(f'results/label_distribution_{title}.png')

if __name__ == "__main__":
    plot_label_distribution("data/processed/data.pt", "train")
    plot_label_distribution("data/processed/test.pt", "test")
