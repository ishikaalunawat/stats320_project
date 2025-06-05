## Data
### 1. Generate file lists
Download .zip data into data/raw/ and unzip it here
Then run: 
```
python src/generate_file_lists.py
```
Note: Make sure to change `RAW_DATA_DIR` and `OUT_FILE` to desired train/test_1/test_2/validation splits.

### 2. Merge file lists
Run (example: `data/train_file_list.txt` and `data/test_file_list.txt`):
```
cat data/train_file_list.txt data/test_file_list.txt > data/merged_file_list.txt
```

### 3. Pre-process and save as tensors
Run:
```
python src/stratified_preprocess.py --file_list data/raw/merged_file_list.txt --train_output data/processed/data.pt --test_output data/processed/test.pt --test_size 0.33
```

### [Optional] Check label distribution in splits
Run to see results in `results/`:
```
python src/check_label_dist.py
```

## Models
You can use any model type: `baseline`, `dropout`, `sparsemax`. Replace the `{model}` tag in the commands below.
### Training
```
python src/train_{model}.py --data data/processed/data.pt --output models/{model}_stratified.pth
```
By default it saves:
* Model in `models/{model}_stratified.pth`
* Train/Val metrics in `results/{model}_stratified/train_metrics.json`

### Evaluation
```
python src/evaluate_{model}.py --data data/processed/test.pt --model models/{model}_stratified.pth --output results/{model}_stratified/test_metrics.json
```
By default it saves:
* Test metrics in `results/{model}_stratified/test_metrics.json`

### Visualizations
#### 1. Loss/Accuracy curves
Run:
```
tensorboard --logdir runs/{model}_stratified
```
#### 2. Confusion Matrices
Run:
```
python src/plot_confusion --model {model}_stratified
```
By default it saves:
* Train/Test confusion matrix plots in `results/{model}_stratified`

