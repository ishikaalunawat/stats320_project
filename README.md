## Data
### Generate file lists
Download .zip data into data/raw/ and unzip it here
Then run: 
```
python src/generate_file_lists.py
```
Note: Make sure to change `RAW_DATA_DIR` and `OUT_FILE` to desired train/test_1/test_2/validation splits.

### Merge file lists
Run (example: `data/train_file_list.txt` and `data/test_file_list.txt`):
```
cat data/train_file_list.txt data/test_file_list.txt > data/merged_file_list.txt
```

### Pre-process and save as tensors
Run:
```
python src/stratified_preprocess.py --file_list data/raw/merged_file_list.txt --train_output data/processed/data.pt --test_output data/processed/test.pt --test_size 0.33
```

### Check label distribution in splits
Run to see results in `results/`:
```
python src/check_label_dist.py
```

## Training
### ResNet-Features (Baseline with 316 features + Dropout):
This model uses dropout. Run to see model saved in `models/`, train metrics in `results/` and tensorboard history in `runs/baseline_dropout_stratified`:
```
python src/train_baseline.py --data data/processed/data.pt --output models/baseline_dropout_stratified.pth
```

### ResNet-SparseMax