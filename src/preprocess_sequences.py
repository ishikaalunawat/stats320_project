import torch
import argparse
import os

def make_sequences(features, labels, window_size=11):
    assert window_size % 2 == 1, "Window size must be odd"
    half = window_size // 2

    N, D = features.shape
    sequences = []
    sequence_labels = []

    for i in range(half, N - half):
        window = features[i - half : i + half + 1]  # shape: [T, D]
        label = labels[i]  # center frame label
        sequences.append(window)
        sequence_labels.append(label)

    # Convert to tensors
    sequences = torch.stack(sequences)         # [N', T, D]
    sequence_labels = torch.tensor(sequence_labels)

    # Transpose to [N', D, T] for Conv1D
    sequences = sequences.permute(0, 2, 1)     # [N', 316, 11]
    return sequences, sequence_labels

def main(args):
    data = torch.load(args.input)
    features, labels = data['features'], data['labels']

    sequences, sequence_labels = make_sequences(features, labels, window_size=args.window)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'features': sequences, 'labels': sequence_labels}, args.output)
    print(f"Saved temporal dataset to: {args.output}")
    print(f"Shape: features {sequences.shape}, labels {sequence_labels.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input data.pt path')
    parser.add_argument('--output', type=str, required=True, help='Output temporal data.pt path')
    parser.add_argument('--window', type=int, default=11, help='Window size (odd number)')
    args = parser.parse_args()
    main(args)
