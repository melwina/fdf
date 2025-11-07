import os, io, json, random, re
import pandas as pd
from zipfile import ZipFile
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Config ----------------
MANIFEST_PATH = "manifest.csv"
MODEL_NAME    = "8multi_face_classifier_xception_early_stopped_intersection_weighted.pth"
RESULTS_DIR   = "results"
RESULTS_FILE  = os.path.join(RESULTS_DIR, "8multi_results_intersection_weighted.txt")
PRED_CSV      = os.path.join(RESULTS_DIR, "multi_predictions_intersection_weighted.csv")

IMG_SIZE    = 299
BATCH_SIZE  = 32
NUM_WORKERS = 4
PIN_MEMORY  = torch.cuda.is_available()

VAL_SAMPLE_N = None

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ---------------- Utils/Transforms ----------------
def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")

def clean_source_key(parent_path: str) -> str:
    parts = [seg for seg in parent_path.split("/") if seg and not re.fullmatch(r"\d+", seg)]
    return "/".join(parts) if parts else "(root)"

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------------- Dataset ----------------
class ManifestEvalDataset(Dataset):
    """
    Returns (image_tensor, multi_label, source_key, image_path_for_output)
    """
    def __init__(self, df: pd.DataFrame, transform=None, label_col="multi_label", failure_log_path="eval_loader_failures.log"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col
        self._zip_handles = {}
        self._pid = os.getpid()
        self._fail_log = failure_log_path

        required_cols = {"zip_path","member", label_col}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {missing}")

    def __len__(self): return len(self.df)

    def _get_zip(self, zip_path: str) -> ZipFile:
        if os.getpid() != self._pid:
            self._zip_handles = {}; self._pid = os.getpid()
        zf = self._zip_handles.get(zip_path)
        if zf is None:
            zf = ZipFile(zip_path, 'r')
            self._zip_handles[zip_path] = zf
        return zf

    def _read_image(self, row) -> Image.Image:
        zip_path = str(row.zip_path) if pd.notna(row.zip_path) else ""
        member   = str(row.member)
        if not zip_path or not zip_path.lower().endswith(".zip"):
            return Image.open(_norm_path(member)).convert("RGB")
        zf = self._get_zip(_norm_path(zip_path))
        with zf.open(member) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row[self.label_col])
        member = str(row.member)
        parent = os.path.dirname(member)
        source_key = clean_source_key(parent)

        # choose a human/audit path for output CSV
        image_path_for_output = row.get("Image Path", None)
        image_path_for_output = str(image_path_for_output) if pd.notna(image_path_for_output) else member

        try:
            img = self._read_image(row)
        except Exception as e:
            with open(self._fail_log, "a") as logf:
                logf.write(json.dumps({"error": str(e), "zip_path": str(row.zip_path), "member": member}) + "\n")
            x = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            return x, label, source_key, image_path_for_output

        if self.transform:
            img = self.transform(img)
        return img, label, source_key, image_path_for_output

    def __del__(self):
        for z in list(self._zip_handles.values()):
            try: z.close()
            except: pass

# ---------------- Load manifest (val-only) ----------------
df = pd.read_csv(MANIFEST_PATH)
df = df[df["val"] == 1].copy()
df = df.drop_duplicates(subset=["Image Path"])
if VAL_SAMPLE_N is not None and len(df) > VAL_SAMPLE_N:
    df = df.sample(n=VAL_SAMPLE_N, random_state=SEED).copy()

print(f"Eval set size (val-only): {len(df)}")

# ---------------- DataLoader ----------------
eval_ds = ManifestEvalDataset(df, transform=val_transform, label_col="multi_label")
eval_loader = DataLoader(
    eval_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=False,
    prefetch_factor=2
)

# ---------------- Model ----------------
class FaceClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x): return self.backbone(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceClassifier(num_classes=4).to(device)
state = torch.load(os.path.join("models", MODEL_NAME), map_location=device)
model.load_state_dict(state)
model.eval()

# ---------------- Evaluate ----------------
class_names = ['deepfake','diffusion','gan','real']
all_preds, all_targets = [], []
all_image_paths = []

source_stats = {}

with torch.inference_mode():
    for batch in eval_loader:
        images, targets, source_keys, img_paths = batch
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images).argmax(1)

        preds_cpu = preds.detach().cpu().numpy()
        t_cpu = targets.detach().cpu().numpy()

        all_preds.extend(preds_cpu.tolist())
        all_targets.extend(t_cpu.tolist())
        all_image_paths.extend(list(img_paths))

        # per-source accum
        for sk, y, yhat in zip(source_keys, t_cpu, preds_cpu):
            st = source_stats.get(sk)
            if st is None:
                st = {"n": 0, "correct": 0, "pred_counts": np.zeros(4, dtype=np.int64)}
                source_stats[sk] = st
            st["n"] += 1
            st["correct"] += int(y == yhat)
            st["pred_counts"][yhat] += 1

# ---------------- Reporting ----------------
os.makedirs(RESULTS_DIR, exist_ok=True)
total = len(all_preds)
acc = (np.array(all_preds) == np.array(all_targets)).mean() if total > 0 else 0.0

with open(RESULTS_FILE, "w") as f:
    f.write(f"Results for {MODEL_NAME} (val-only)\n")
    f.write(f"Evaluated samples: {total}\n")
    f.write(f"Overall accuracy: {acc*100:.2f}%\n\n")

    pred_counts = np.bincount(np.array(all_preds), minlength=4) if total else np.zeros(4, dtype=np.int64)
    f.write("Global predicted distribution:\n")
    for cid, cnt in enumerate(pred_counts):
        pct = (cnt/total*100) if total else 0.0
        f.write(f"  Class {cid} ({class_names[cid]}): {cnt} ({pct:.1f}%)\n")
    f.write("\n")

    try:
        f.write("Classification report (true vs pred):\n")
        f.write(classification_report(all_targets, all_preds, target_names=class_names))
    except Exception as e:
        f.write(f"(Could not compute classification_report: {e})\n")
    f.write("\n")

    try:
        cm = confusion_matrix(all_targets, all_preds, labels=[0,1,2,3])
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))
        f.write("\n\n")
    except Exception as e:
        f.write(f"(Could not compute confusion_matrix: {e})\n\n")

    f.write("Per-source breakdown (grouped by parent path of `member`, numeric-only dirs removed):\n")
    for sk in sorted(source_stats.keys()):
        st = source_stats[sk]
        n = st["n"]
        acc_s = (st["correct"]/n*100) if n else 0.0
        f.write(f"\nSource: {sk}\n")
        f.write(f"  n={n}, accuracy={acc_s:.2f}%\n")
        f.write("  Predicted distribution:\n")
        for cid in range(4):
            cnt = int(st["pred_counts"][cid])
            pct = (cnt/n*100) if n else 0.0
            f.write(f"    → {cid} ({class_names[cid]}): {cnt} ({pct:.1f}%)\n")

print(f"Wrote evaluation summary with per-source grouping to: {RESULTS_FILE}")

# ---------------- Save per-image predictions CSV ----------------
pred_df = pd.DataFrame({
    "Image Path": all_image_paths,   # from manifest Image Path if present, else member
    "true": all_targets,
    "pred": all_preds
})
pred_df.to_csv(PRED_CSV, index=False)
print(f"Saved {len(pred_df)} per-image predictions to: {PRED_CSV}")
