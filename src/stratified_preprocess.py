import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from data_reader import load_features_and_annotations_from_file_list, convert_features_annotation_dictionary_to_list, label2id

'''
It's different from the preprocess.py script in that it performs to
make sure that the label distribution is preserved in both training and test sets.
(since test_1 and test_2 have no 'mount' or class idx '2' samples)
'''

def clean_and_normalize(features):
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -1e6, 1e6)
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def label_str_to_id(labels):
    return [label2id.get(label, 0) for label in labels]

def save_to_tensor(features, labels, output_path):
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'features': features_tensor, 'labels': labels_tensor}, output_path)
    print(f"Saved to {output_path} with shape: {features_tensor.shape}, {labels_tensor.shape}")

def main(args):
    # Step 1: Load and parse data
    file_list = [line.strip() for line in open(args.file_list)]
    data_dict = load_features_and_annotations_from_file_list(file_list)
    print(f"Loaded {len(data_dict)} valid samples from {len(file_list)} files")
    feature_list, annotation_list = convert_features_annotation_dictionary_to_list(data_dict)

    all_features = []
    all_labels = []

    for feat, labels in zip(feature_list, annotation_list):
        if len(feat) != len(labels):
            min_len = min(len(feat), len(labels))
            feat = feat[:min_len]
            labels = labels[:min_len]
        feat = clean_and_normalize(feat)
        labels = label_str_to_id(labels)
        all_features.append(feat)
        all_labels.append(labels)

    # Step 2: Flatten into single dataset
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("Total samples:", len(all_labels))
    print("Label distribution:", dict(Counter(all_labels)))

    # Step 3: Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=args.test_size,
        stratify=all_labels, random_state=42
    )

    print("Train label distribution:", dict(Counter(y_train)))
    print("Test label distribution:", dict(Counter(y_test)))

    # Step 4: Save
    save_to_tensor(X_train, y_train, args.train_output)
    save_to_tensor(X_test, y_test, args.test_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=str, required=True, help='File listing pose|annot|feat paths')
    parser.add_argument('--train_output', type=str, default='data/processed/data.pt')
    parser.add_argument('--test_output', type=str, default='data/processed/test.pt')
    parser.add_argument('--test_size', type=float, default=0.33, help='Fraction for test set')
    args = parser.parse_args()
    main(args)