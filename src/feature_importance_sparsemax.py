import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import argparse
import json
from train_sparsemax import ResNetClassifier
import matplotlib.pyplot as plt

names = np.array([b'nose_x', b'nose_y', b'right_ear_x', b'right_ear_y',
                  b'left_ear_x', b'left_ear_y', b'neck_x', b'neck_y',
                  b'right_side_x', b'right_side_y', b'left_side_x', b'left_side_y',
                  b'tail_base_x', b'tail_base_y', b'centroid_x', b'centroid_y',
                  b'centroid_head_x', b'centroid_head_y', b'centroid_hips_x',
                  b'centroid_hips_y', b'centroid_body_x', b'centroid_body_y', b'phi',
                  b'ori_head', b'ori_body', b'angle_head_body_l',
                  b'angle_head_body_r', b'major_axis_len', b'minor_axis_len',
                  b'axis_ratio', b'area_ellipse', b'dist_edge_x', b'dist_edge_y',
                  b'dist_edge', b'speed', b'speed_centroid', b'acceleration',
                  b'acceleration_centroid', b'speed_fwd', b'resh_twd_itrhb',
                  b'pixel_change_ubbox_mice', b'pixel_change', b'nose_pc',
                  b'right_ear_pc', b'left_ear_pc', b'neck_pc', b'right_side_pc',
                  b'left_side_pc', b'tail_base_pc', b'rel_angle_social',
                  b'rel_dist_gap', b'rel_dist_scaled', b'rel_dist_centroid',
                  b'rel_dist_nose', b'rel_dist_head', b'rel_dist_body',
                  b'rel_dist_head_body', b'rel_dist_centroid_change',
                  b'overlap_bboxes', b'area_ellipse_ratio', b'angle_between',
                  b'facing_angle', b'radial_vel', b'tangential_vel',
                  b'dist_m1nose_m2nose', b'dist_m1nose_m2right_ear',
                  b'dist_m1nose_m2left_ear', b'dist_m1nose_m2neck',
                  b'dist_m1nose_m2right_side', b'dist_m1nose_m2left_side',
                  b'dist_m1nose_m2tail_base', b'dist_m1right_ear_m2nose',
                  b'dist_m1right_ear_m2right_ear', b'dist_m1right_ear_m2left_ear',
                  b'dist_m1right_ear_m2neck', b'dist_m1right_ear_m2right_side',
                  b'dist_m1right_ear_m2left_side', b'dist_m1right_ear_m2tail_base',
                  b'dist_m1left_ear_m2nose', b'dist_m1left_ear_m2right_ear',
                  b'dist_m1left_ear_m2left_ear', b'dist_m1left_ear_m2neck',
                  b'dist_m1left_ear_m2right_side', b'dist_m1left_ear_m2left_side',
                  b'dist_m1left_ear_m2tail_base', b'dist_m1neck_m2nose',
                  b'dist_m1neck_m2right_ear', b'dist_m1neck_m2left_ear',
                  b'dist_m1neck_m2neck', b'dist_m1neck_m2right_side',
                  b'dist_m1neck_m2left_side', b'dist_m1neck_m2tail_base',
                  b'dist_m1right_side_m2nose', b'dist_m1right_side_m2right_ear',
                  b'dist_m1right_side_m2left_ear', b'dist_m1right_side_m2neck',
                  b'dist_m1right_side_m2right_side',
                  b'dist_m1right_side_m2left_side', b'dist_m1right_side_m2tail_base',
                  b'dist_m1left_side_m2nose', b'dist_m1left_side_m2right_ear',
                  b'dist_m1left_side_m2left_ear', b'dist_m1left_side_m2neck',
                  b'dist_m1left_side_m2right_side', b'dist_m1left_side_m2left_side',
                  b'dist_m1left_side_m2tail_base', b'dist_m1tail_base_m2nose',
                  b'dist_m1tail_base_m2right_ear', b'dist_m1tail_base_m2left_ear',
                  b'dist_m1tail_base_m2neck', b'dist_m1tail_base_m2right_side',
                  b'dist_m1tail_base_m2left_side', b'dist_m1tail_base_m2tail_base',
                  b'dist_nose_right_ear', b'dist_nose_left_ear', b'dist_nose_neck',
                  b'dist_nose_right_side', b'dist_nose_left_side',
                  b'dist_nose_tail_base', b'dist_right_ear_left_ear',
                  b'dist_right_ear_neck', b'dist_right_ear_right_side',
                  b'dist_right_ear_left_side', b'dist_right_ear_tail_base',
                  b'dist_left_ear_neck', b'dist_left_ear_right_side',
                  b'dist_left_ear_left_side', b'dist_left_ear_tail_base',
                  b'dist_neck_right_side', b'dist_neck_left_side',
                  b'dist_neck_tail_base', b'dist_right_side_left_side',
                  b'dist_right_side_tail_base', b'dist_left_side_tail_base',
                  b'speed_centroid_w2', b'speed_centroid_w5', b'speed_centroid_w10',
                  b'speed_nose_w2', b'speed_nose_w5', b'speed_nose_w10',
                  b'speed_right_ear_w2', b'speed_right_ear_w5',
                  b'speed_right_ear_w10', b'speed_left_ear_w2', b'speed_left_ear_w5',
                  b'speed_left_ear_w10', b'speed_neck_w2', b'speed_neck_w5',
                  b'speed_neck_w10', b'speed_right_side_w2', b'speed_right_side_w5',
                  b'speed_right_side_w10', b'speed_left_side_w2',
                  b'speed_left_side_w5', b'speed_left_side_w10',
                  b'speed_tail_base_w2', b'speed_tail_base_w5',
                  b'speed_tail_base_w10'])

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

def plot_deltas(deltas, plot_path):
  '''
  Plot a histogram for the mean drop in accuracy across feature shuffles.
  '''
  plt.hist(deltas)
  plt.xlabel('Mean Drop in Accuracy')
  plt.ylabel('Number of Features')

  plt.savefig(plot_path)

def feature_importance(data_path, model_path, metric_path, output_path, 
                       plot_path, n_shuffles=10):
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

    sorted, rankings = mean_deltas.sort(descending=True)
        
    with open(output_path, 'w') as f:
        json.dump({'deltas': sorted, 'rankings': names[rankings]}, f)

    plot_deltas(sorted, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/test.pt')
    parser.add_argument('--model', type=str,  default='models/sparsemax_stratified.pth')
    parser.add_argument('--metric', type=str, default='results/sparsemax_stratified/test_metrics.json')
    parser.add_argument('--output', type=str, default='results/sparsemax_stratified/feature_importance.json')
    parser.add_argument('--plot', type=str, default='results/sparsemax_stratified/feature_importance.png')
    parser.add_argument('--shuffles', type=int, default=10)
    args = parser.parse_args()

    feature_importance(args.data, args.model, args.metric, args.output, args.plot, args.shuffles)
