import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import argparse
import json
from train_sparsemax import ResNetClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

names = np.array(['nose_x', 'nose_y', 'right_ear_x', 'right_ear_y',
                  'left_ear_x', 'left_ear_y', 'neck_x', 'neck_y',
                  'right_side_x', 'right_side_y', 'left_side_x', 'left_side_y',
                  'tail_base_x', 'tail_base_y', 'centroid_x', 'centroid_y',
                  'centroid_head_x', 'centroid_head_y', 'centroid_hips_x',
                  'centroid_hips_y', 'centroid_body_x', 'centroid_body_y', 'phi',
                  'ori_head', 'ori_body', 'angle_head_body_l',
                  'angle_head_body_r', 'major_axis_len', 'minor_axis_len',
                  'axis_ratio', 'area_ellipse', 'dist_edge_x', 'dist_edge_y',
                  'dist_edge', 'speed', 'speed_centroid', 'acceleration',
                  'acceleration_centroid', 'speed_fwd', 'resh_twd_itrhb',
                  'pixel_change_ubbox_mice', 'pixel_change', 'nose_pc',
                  'right_ear_pc', 'left_ear_pc', 'neck_pc', 'right_side_pc',
                  'left_side_pc', 'tail_base_pc', 'rel_angle_social',
                  'rel_dist_gap', 'rel_dist_scaled', 'rel_dist_centroid',
                  'rel_dist_nose', 'rel_dist_head', 'rel_dist_body',
                  'rel_dist_head_body', 'rel_dist_centroid_change',
                  'overlap_bboxes', 'area_ellipse_ratio', 'angle_between',
                  'facing_angle', 'radial_vel', 'tangential_vel',
                  'dist_m1nose_m2nose', 'dist_m1nose_m2right_ear',
                  'dist_m1nose_m2left_ear', 'dist_m1nose_m2neck',
                  'dist_m1nose_m2right_side', 'dist_m1nose_m2left_side',
                  'dist_m1nose_m2tail_base', 'dist_m1right_ear_m2nose',
                  'dist_m1right_ear_m2right_ear', 'dist_m1right_ear_m2left_ear',
                  'dist_m1right_ear_m2neck', 'dist_m1right_ear_m2right_side',
                  'dist_m1right_ear_m2left_side', 'dist_m1right_ear_m2tail_base',
                  'dist_m1left_ear_m2nose', 'dist_m1left_ear_m2right_ear',
                  'dist_m1left_ear_m2left_ear', 'dist_m1left_ear_m2neck',
                  'dist_m1left_ear_m2right_side', 'dist_m1left_ear_m2left_side',
                  'dist_m1left_ear_m2tail_base', 'dist_m1neck_m2nose',
                  'dist_m1neck_m2right_ear', 'dist_m1neck_m2left_ear',
                  'dist_m1neck_m2neck', 'dist_m1neck_m2right_side',
                  'dist_m1neck_m2left_side', 'dist_m1neck_m2tail_base',
                  'dist_m1right_side_m2nose', 'dist_m1right_side_m2right_ear',
                  'dist_m1right_side_m2left_ear', 'dist_m1right_side_m2neck',
                  'dist_m1right_side_m2right_side',
                  'dist_m1right_side_m2left_side', 'dist_m1right_side_m2tail_base',
                  'dist_m1left_side_m2nose', 'dist_m1left_side_m2right_ear',
                  'dist_m1left_side_m2left_ear', 'dist_m1left_side_m2neck',
                  'dist_m1left_side_m2right_side', 'dist_m1left_side_m2left_side',
                  'dist_m1left_side_m2tail_base', 'dist_m1tail_base_m2nose',
                  'dist_m1tail_base_m2right_ear', 'dist_m1tail_base_m2left_ear',
                  'dist_m1tail_base_m2neck', 'dist_m1tail_base_m2right_side',
                  'dist_m1tail_base_m2left_side', 'dist_m1tail_base_m2tail_base',
                  'dist_nose_right_ear', 'dist_nose_left_ear', 'dist_nose_neck',
                  'dist_nose_right_side', 'dist_nose_left_side',
                  'dist_nose_tail_base', 'dist_right_ear_left_ear',
                  'dist_right_ear_neck', 'dist_right_ear_right_side',
                  'dist_right_ear_left_side', 'dist_right_ear_tail_base',
                  'dist_left_ear_neck', 'dist_left_ear_right_side',
                  'dist_left_ear_left_side', 'dist_left_ear_tail_base',
                  'dist_neck_right_side', 'dist_neck_left_side',
                  'dist_neck_tail_base', 'dist_right_side_left_side',
                  'dist_right_side_tail_base', 'dist_left_side_tail_base',
                  'speed_centroid_w2', 'speed_centroid_w5', 'speed_centroid_w10',
                  'speed_nose_w2', 'speed_nose_w5', 'speed_nose_w10',
                  'speed_right_ear_w2', 'speed_right_ear_w5',
                  'speed_right_ear_w10', 'speed_left_ear_w2', 'speed_left_ear_w5',
                  'speed_left_ear_w10', 'speed_neck_w2', 'speed_neck_w5',
                  'speed_neck_w10', 'speed_right_side_w2', 'speed_right_side_w5',
                  'speed_right_side_w10', 'speed_left_side_w2',
                  'speed_left_side_w5', 'speed_left_side_w10',
                  'speed_tail_base_w2', 'speed_tail_base_w5',
                  'speed_tail_base_w10'])

# prefixes for two mice
m1_names = ['m1_' + name for name in names]
m2_names = ['m2_' + name for name in names]

# full 316-feature list
full_names = np.array(m1_names + m2_names)

def evaluate_model(x, y, model, device):
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

def plot_top_k_features(deltas, names, plot_path, top_k=15):
    topk_vals, topk_indices = torch.topk(deltas, top_k)
    topk_names = names[topk_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), topk_vals.numpy()[::-1], color='steelblue')
    plt.yticks(range(top_k), topk_names[::-1])
    plt.xlabel('Mean Drop in Accuracy')
    plt.title(f'Top {top_k} Most Important Features')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def feature_importance(data_path, model_path, metric_path, output_path, 
                       plot_path, n_shuffles=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # accuracy before shuffling features
    with open(metric_path, 'r') as f:
        initial_accuracy = json.load(f)['accuracy']


    # test data
    data = torch.load(data_path)
    x, y = data['features'], data['labels']


    # load model
    model = ResNetClassifier(input_dim=x.shape[1], num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()


    # shuffle
    n_samples, n_features = x.shape

    # to store mean drop for each feature
    mean_deltas = torch.zeros(n_features)
    
    for i in range(n_features):
        print(f"Processing feature {i}:", full_names[i])
        # drop for each shuffle
        deltas = torch.zeros(n_shuffles)

        for j in range(n_shuffles):
            perm = torch.randperm(n_samples)
            new_data = x.detach().clone()
            new_data[:, i] = new_data[perm, i]

            deltas[j] = initial_accuracy - evaluate_model(new_data, y, model, device)

        mean_deltas[i] = deltas.mean()

    sorted, rankings = mean_deltas.sort(descending=True)
        
    with open(output_path, 'w') as f:
        json.dump({'deltas': sorted.tolist(), 'rankings': full_names[rankings].tolist()}, f)

    plot_top_k_features(mean_deltas, full_names, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/test.pt')
    parser.add_argument('--model', type=str,  default='models/sparsemax_stratified.pth')
    parser.add_argument('--metric', type=str, default='results/sparsemax_stratified/test_metrics.json')
    parser.add_argument('--output', type=str, default='results/sparsemax_stratified/feature_importance.json')
    parser.add_argument('--plot', type=str, default='results/sparsemax_stratified/topk_importance.png')
    parser.add_argument('--shuffles', type=int, default=10)
    args = parser.parse_args()

    feature_importance(args.data, args.model, args.metric, args.output, args.plot, args.shuffles)
