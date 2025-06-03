import os
import glob

RAW_DATA_DIR = "data/raw/test_1"
OUT_FILE = "data/test_file_list.txt"
f = open(OUT_FILE, "x")

with open(OUT_FILE, 'w') as f_out:
    for session in os.listdir(RAW_DATA_DIR):
        session_dir = os.path.join(RAW_DATA_DIR, session)
        if not os.path.isdir(session_dir):
            continue

        # Search for required files
        pose_files = glob.glob(os.path.join(session_dir, "*_pose_top_v1_*.json"))
        feat_files = glob.glob(os.path.join(session_dir, "*_raw_feat_top_v1_*.npz"))
        annot_files = glob.glob(os.path.join(session_dir, "*.annot"))

        if pose_files and annot_files and feat_files:
            line = f"{pose_files[0]}|{annot_files[0]}|{feat_files[0]}\n"
            f_out.write(line)
        else:
            print(f"Missing file in {session_dir}")
