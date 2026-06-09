import os
import sys
import argparse
import zipfile
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

# PyTorch float8 compatibility fix
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# NumPy 2.0 compatibility patch for TrackEval (older dependencies use np.float/np.int/np.bool)
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

# Get project roots and paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(CURRENT_DIR, 'model', 'football-player-detection.pt')

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.buffer = []
        self.saved = False

    def write(self, message):
        self.terminal.write(message)
        self.buffer.append(message)

    def flush(self):
        self.terminal.flush()

    def save_with_summary_at_top(self, results, split, tracker_name):
        if self.saved:
            return
        self.saved = True
        
        # Construct summary table string
        summary_lines = []
        summary_lines.append("="*50)
        summary_lines.append("          SOCCERNET EVALUATION METRICS")
        summary_lines.append("="*50)
        summary_lines.append(f" Split: {split.upper()}")
        summary_lines.append(f" Tracker: {tracker_name}")
        summary_lines.append("-"*50)
        if results:
            for metric, val in results.items():
                summary_lines.append(f"  {metric:<15} : {val*100:.2f}%")
        else:
            summary_lines.append("  No metrics available (evaluation failed or was skipped).")
        summary_lines.append("="*50)
        summary_str = "\n".join(summary_lines) + "\n\n"
        
        full_log = "".join(self.buffer)
        
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(summary_str)
            f.write("="*50 + "\n")
            f.write("          DETAILED RUN & TRACKEVAL LOGS\n")
            f.write("="*50 + "\n\n")
            f.write(full_log)



def extract_zip(zip_path, extract_to):
    print(f"[*] Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[+] Extraction complete.")


def zip_predictions(pred_data_dir, zip_out_path):
    print(f"[*] Zipping predictions from {pred_data_dir} to {zip_out_path}...")
    with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(pred_data_dir):
            for file in files:
                if file.endswith('.txt'):
                    zipf.write(os.path.join(root, file), arcname=file)
    print(f"[+] Zipped predictions successfully.")


def write_mock_gt_files(seq_dest, seq_name):
    # Create empty gt.txt
    with open(os.path.join(seq_dest, "gt", "gt.txt"), "w") as f:
        pass
    # Create mock seqinfo.ini
    mock_ini = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate=25
seqLength=1
imWidth=1920
imHeight=1080
imExt=.jpg
"""
    with open(os.path.join(seq_dest, "seqinfo.ini"), "w") as f:
        f.write(mock_ini)


def construct_gt_zip(seq_dirs, zip_out_path, split, expected_seqs):
    print(f"[*] Constructing ground truth zip from sequence directories at {zip_out_path}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create <split>-evalAI folder expected by SoccerNet.Evaluation.Tracking
        eval_ai_dir = os.path.join(temp_dir, f"{split}-evalAI")
        os.makedirs(eval_ai_dir, exist_ok=True)
        
        # Build lookup for sequences we actually have
        local_seqs = {os.path.basename(p): p for p in seq_dirs}
        
        valid_seqs = 0
        for seq_name in expected_seqs:
            seq_dest = os.path.join(eval_ai_dir, seq_name)
            os.makedirs(os.path.join(seq_dest, "gt"), exist_ok=True)
            
            if seq_name in local_seqs:
                seq_path = local_seqs[seq_name]
                gt_txt_src = os.path.join(seq_path, "gt", "gt.txt")
                seqinfo_src = os.path.join(seq_path, "seqinfo.ini")
                
                if os.path.exists(gt_txt_src) and os.path.exists(seqinfo_src):
                    shutil.copy2(gt_txt_src, os.path.join(seq_dest, "gt", "gt.txt"))
                    shutil.copy2(seqinfo_src, os.path.join(seq_dest, "seqinfo.ini"))
                    valid_seqs += 1
                else:
                    write_mock_gt_files(seq_dest, seq_name)
            else:
                write_mock_gt_files(seq_dest, seq_name)
                
        # Zip the contents of temp_dir
        with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname=arcname)
                    
    print(f"[+] Ground truth zip with {valid_seqs} real and {len(expected_seqs)-valid_seqs} mock sequences constructed.")
    return True


def run_tracking_evaluation(args):
    # Map 'gpu' to 'cuda' to prevent PyTorch device string issues
    if args.device.lower() == 'gpu':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[*] Mapped '--device gpu' to '{args.device}' based on CUDA availability.")
    print(f"[*] Running evaluation on device: {args.device}")

    # 1. Dataset Downloading & Extraction
    if args.download_labels or args.download_data:
        print("[*] Initializing SoccerNet Downloader...")
        downloader = SoccerNetDownloader(LocalDirectory=args.dataset_dir)
        
        if args.download_labels:
            print(f"[*] Downloading labels for split: {args.split}...")
            try:
                downloader.downloadDataTask(task="tracking", split=[f"{args.split}_labels"])
            except Exception as e:
                print(f"[!] Failed to download official labels zip: {e}")
                print("[*] Local evaluation will try to construct labels dynamically from downloaded sequence files.")
            
        if args.download_data:
            print(f"[*] Downloading visual data for split: {args.split}...")
            downloader.downloadDataTask(task="tracking", split=[args.split])
            
            # Extract data zip if it exists
            data_zip = os.path.join(args.dataset_dir, "tracking", f"{args.split}.zip")
            if os.path.exists(data_zip):
                extract_to = os.path.join(args.dataset_dir, "tracking", args.split)
                extract_zip(data_zip, extract_to)

    # 2. Locating sequence directories
    seq_dirs = []
    for root, dirs, files in os.walk(args.dataset_dir):
        if "img1" in dirs and "seqinfo.ini" in files:
            seq_dirs.append(root)
            
    seq_dirs = sorted(seq_dirs)
    
    if not seq_dirs:
        print(f"[!] No sequence directories containing 'img1' and 'seqinfo.ini' found in '{args.dataset_dir}'")
        print("[!] If you have downloaded the zip file, make sure it is extracted.")
        print("[!] Example structure: data/SoccerNet/tracking/tracking/test/<seq_name>/img1/")
        return
        
    if args.limit_seqs:
        seq_dirs = seq_dirs[:args.limit_seqs]
        print(f"[*] Limiting processing to first {len(seq_dirs)} sequences.")
        
    print(f"[*] Found {len(seq_dirs)} local sequences to track.")

    # 3. Read expected sequences from split map
    map_file = os.path.join(os.path.dirname(SoccerNet.__file__), "data", f"SNMOT-{args.split}.txt")
    expected_seqs = []
    if os.path.exists(map_file):
        with open(map_file, "r") as f:
            expected_seqs = [line.strip() for line in f if line.strip() and line.strip() != "name"]
        print(f"[*] Loaded sequence map file: {os.path.basename(map_file)} ({len(expected_seqs)} expected sequences)")
    else:
        # Fallback to local seqs if map file doesn't exist
        expected_seqs = [os.path.basename(p) for p in seq_dirs]
        print(f"[!] Sequence map file SNMOT-{args.split}.txt not found. Using {len(expected_seqs)} local sequences.")

    # 4. Create prediction output directories
    pred_data_dir = os.path.join("benchmarks", args.tracker_name, "data")
    os.makedirs(pred_data_dir, exist_ok=True)

    # 5. Load player detection model and run tracking loop
    if not args.skip_tracking:
        print(f"[*] Loading player detection model from {PLAYER_DETECTION_MODEL_PATH}...")
        if not os.path.exists(PLAYER_DETECTION_MODEL_PATH):
            print(f"[!] Model file not found at: {PLAYER_DETECTION_MODEL_PATH}")
            print("[!] Please run setup.bat or setup.sh to download models first.")
            return
            
        player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=args.device)

        # 6. Tracking Loop
        print("[*] Running tracking on all sequences...")
        for seq_path in seq_dirs:
            seq_name = os.path.basename(seq_path)
            print(f"\n--- Processing: {seq_name} ---")
            
            # Initialize fresh ByteTrack tracker for each sequence
            tracker = sv.ByteTrack(minimum_consecutive_frames=3)
            
            img_dir = os.path.join(seq_path, "img1")
            images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            out_file_path = os.path.join(pred_data_dir, f"{seq_name}.txt")
            with open(out_file_path, "w") as out_file:
                for frame_idx, img_path in enumerate(tqdm(images, desc=f"Tracking {seq_name}"), start=1):
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    
                    # YOLOv8 inference
                    result = player_model(frame, imgsz=args.img_size, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(result)
                    detections = tracker.update_with_detections(detections)
                    
                    # Write in MOT challenge format
                    if detections.tracker_id is not None:
                        for xyxy, confidence, tracker_id in zip(detections.xyxy, detections.confidence, detections.tracker_id):
                            x1, y1, x2, y2 = xyxy
                            w = x2 - x1
                            h = y2 - y1
                            # frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                            out_file.write(f"{frame_idx},{tracker_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{confidence:.4f},-1,-1,-1\n")
                            
            print(f"[+] Saved predictions to: {out_file_path}")
    else:
        print("[*] Skipping tracking step. Evaluating existing predictions in benchmarks folder...")

    # Write empty files for any expected sequences that we did not run
    missing_seqs = 0
    for seq_name in expected_seqs:
        pred_file = os.path.join(pred_data_dir, f"{seq_name}.txt")
        if not os.path.exists(pred_file):
            with open(pred_file, "w") as f:
                pass
            missing_seqs += 1
    if missing_seqs > 0:
        print(f"[*] Created empty prediction files for {missing_seqs} missing expected sequences.")

    # 7. Zipping Predictions
    print("\n[*] Preparing prediction zip file for SoccerNet evaluation...")
    pred_zip_path = os.path.join("benchmarks", f"soccernet_mot_results.zip")
    zip_predictions(pred_data_dir, pred_zip_path)
    
    # Locate or construct ground truth labels zip
    gt_zip = args.gt_zip
    if not gt_zip:
        default_gt_zip = os.path.join(args.dataset_dir, "tracking", f"{args.split}_labels.zip")
        if os.path.exists(default_gt_zip):
            gt_zip = default_gt_zip
        else:
            # Search recursively under dataset_dir for f"{split}_labels.zip"
            for root, dirs, files in os.walk(args.dataset_dir):
                for f in files:
                    if f.endswith(f"{args.split}_labels.zip"):
                        gt_zip = os.path.join(root, f)
                        break
                if gt_zip:
                    break
                    
    # Fallback: construct GT zip dynamically
    if not gt_zip or not os.path.exists(gt_zip):
        print("[!] Official ground truth zip file not found.")
        temp_gt_zip = os.path.join("benchmarks", f"temp_{args.split}_gt.zip")
        success = construct_gt_zip(seq_dirs, temp_gt_zip, args.split, expected_seqs)
        if success:
            gt_zip = temp_gt_zip

    if not gt_zip or not os.path.exists(gt_zip):
        print(f"\n[!] Ground truth labels not available.")
        print("[!] Local metric evaluation skipped. Run with --download_labels to get GT.")
        return

    print(f"[*] Using ground truth zip: {gt_zip}")
    print("[*] Running TrackEval evaluation on SoccerNet benchmark...")
    
    # Patch sys.argv to hide our script arguments from SoccerNet's internal argparse
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    try:
        from SoccerNet.Evaluation.Tracking import evaluate
        results = evaluate(gt_zip, pred_zip_path, split=args.split)
        
        # Display Results
        print("\n" + "="*50)
        print("          SOCCERNET EVALUATION METRICS")
        print("="*50)
        print(f" Split: {args.split.upper()}")
        print(f" Tracker: {args.tracker_name}")
        print("-"*50)
        for metric, val in results.items():
            print(f"  {metric:<15} : {val*100:.2f}%")
        print("="*50 + "\n")
        
        # Save to file with summary at the top
        if isinstance(sys.stdout, Logger):
            sys.stdout.save_with_summary_at_top(results, args.split, args.tracker_name)
        
        # Cleanup temp constructed zip if created
        if gt_zip.endswith(f"temp_{args.split}_gt.zip") and os.path.exists(gt_zip):
            os.remove(gt_zip)
            
    except ImportError as e:
        print(f"\n[!] Could not run evaluation: {e}")
        print("[!] Please make sure trackeval is installed via:")
        print("    pip install git+https://github.com/JonathonLuiten/TrackEval.git")
    finally:
        # Restore sys.argv
        sys.argv = original_argv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SoccerNet Tracking Benchmark Evaluation')
    parser.add_argument('--dataset_dir', type=str, default='data/SoccerNet/tracking',
                        help='Path to SoccerNet dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test', 'challenge'],
                        help='Dataset split to evaluate')
    parser.add_argument('--download_labels', action='store_true',
                        help='Download tracking ground-truth labels zip')
    parser.add_argument('--download_data', action='store_true',
                        help='Download tracking visual video frames zip')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
                        help='Hardware device (cpu, cuda, mps, gpu)')
    parser.add_argument('--tracker_name', type=str, default='ByteTrack',
                        help='Folder/Tracker name for outputting results')
    parser.add_argument('--img_size', type=int, default=1280,
                        help='Inference image size for YOLOv8 (default: 1280)')
    parser.add_argument('--limit_seqs', type=int, default=None,
                        help='Limit evaluation to first N sequences (for quick test)')
    parser.add_argument('--gt_zip', type=str, default=None,
                        help='Explicit path to ground-truth labels zip file')
    parser.add_argument('--skip_tracking', action='store_true',
                        help='Skip inference tracking loop and evaluate existing predictions directly')
                        
    args = parser.parse_args()
    
    # Set up dual logger to automatically save terminal output to benchmarks/evaluation_results.txt
    os.makedirs("benchmarks", exist_ok=True)
    log_path = os.path.join("benchmarks", "evaluation_results.txt")
    sys.stdout = Logger(log_path)
    
    try:
        run_tracking_evaluation(args)
    finally:
        if isinstance(sys.stdout, Logger):
            sys.stdout.save_with_summary_at_top(None, args.split, args.tracker_name)
            sys.stdout = sys.stdout.terminal
