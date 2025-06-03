import os
import argparse
import numpy as np
import torch
from data_reader import load_features_and_annotations_from_file_list, convert_features_annotation_dictionary_to_list
from sklearn.preprocessing import StandardScaler

def clean_and_normalize(features):
    # change NaN and Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # clip extreme values
    features = np.clip(features, -1e6, 1e6)

    # normalize extracted features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

def save_to_tensor(features, labels, output_path):
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'features': features_tensor, 'labels': labels_tensor}, output_path)
    print(f"Saved preprocessed data to {output_path}")

def label_str_to_id(labels):
    from data_reader import label2id
    return [label2id.get(label, 0) for label in labels]  # default to 0 = 'other'

def main(args):
    # load raw feature and annotation files
    file_list = [line.strip() for line in open(args.file_list)]  # each line: pose.json|annot.txt|features.npz
    data_dict = load_features_and_annotations_from_file_list(file_list)

    # dictionary to list of features and annotations
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

    # concat all sequences
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # save
    save_to_tensor(all_features, all_labels, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=str, required=True, help='Path to text file listing paths: pose|annot|feature')
    parser.add_argument('--output', type=str, default='data/processed/data.pt', help='Path to save preprocessed data')
    args = parser.parse_args()
    main(args)
