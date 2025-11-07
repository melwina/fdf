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
TEST_PATH = "testset_annotations.csv"
ZIP_PATH = "ai_face_competition_testset.zip"
MODEL_NAME    = "8multi_face_classifier_xception_early_stopped_intersection_weighted.pth"
RESULTS_DIR   = "results"
RESULTS_FILE  = os.path.join(RESULTS_DIR, f"{MODEL_NAME[:1]}_multi_results.txt")
PRED_CSV      = os.path.join(RESULTS_DIR, f"{MODEL_NAME[:1]}_multi_predictions.csv")

IMG_SIZE    = 299
BATCH_SIZE  = 256
NUM_WORKERS = 8
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
    Returns (image_tensor, source_key, image_path_for_output)
    """
    def __init__(self, df: pd.DataFrame, transform=None, failure_log_path="eval_loader_failures.log"):
        if "Image Path" not in df.columns:
            raise ValueError("manifest missing column: {'Image Path'}")
        self.df = df.reset_index(drop=True).copy()
        self.df["member"] = self.df["Image Path"].astype(str).str.lstrip("/")  # internal path inside zip
        self.transform = transform
        self._fail_log = failure_log_path
        self._zip = None
        self._pid = os.getpid()

    def __len__(self):
        return len(self.df)

    def _get_zip(self) -> ZipFile:
        # Reopen if forked to new process or not yet opened
        if self._zip is None or os.getpid() != self._pid:
            self._zip = ZipFile(ZIP_PATH, "r")
            self._pid = os.getpid()
        return self._zip

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        member = row["member"]
        image_path_for_output = row["Image Path"]
        source_key = os.path.dirname(member)

        try:
            zf = self._get_zip()
            with zf.open(member) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception as e:
            with open(self._fail_log, "a") as logf:
                logf.write(json.dumps({"error": str(e), "zip_path": ZIP_PATH, "member": member,
                                       "image_path": str(image_path_for_output)}) + "\n")
            x = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            return x, source_key, image_path_for_output

        if self.transform:
            img = self.transform(img)
        return img, source_key, image_path_for_output

    def __del__(self):
        try:
            if self._zip is not None:
                self._zip.close()
        except:
            pass
# ---------------- Load manifest (val-only) ----------------
df = pd.read_csv(TEST_PATH)
if VAL_SAMPLE_N is not None and len(df) > VAL_SAMPLE_N:
    df = df.sample(n=VAL_SAMPLE_N, random_state=SEED).copy()

print(f"Test set size: {len(df)}")

# ---------------- DataLoader ----------------
eval_ds = ManifestEvalDataset(df, transform=val_transform)
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
all_preds = []
all_image_paths = []

from tqdm import tqdm

pbar = tqdm(total=len(eval_ds), desc="Evaluating", unit="img")
with torch.inference_mode():
    for batch in eval_loader:
        images, source_keys, img_paths = batch
        images = images.to(device, non_blocking=True)

        preds = model(images).argmax(1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_image_paths.extend(list(img_paths))

        pbar.update(len(images))
pbar.close()

# ---------------- Save per-image predictions CSV ----------------
pred_df = pd.DataFrame({
    "Image Path": all_image_paths,   # from manifest Image Path if present, else member
    "multi_pred": all_preds,
    "pred": [0 if p==3 else 1 for p in all_preds],
})
pred_df.to_csv(PRED_CSV, index=False)
print(f"Saved {len(pred_df)} per-image predictions to: {PRED_CSV}")
#save as txt file
pred_df.to_csv(PRED_CSV[:-4]+".txt", index=False, sep="\t")
print(f"Saved {len(pred_df)} per-image predictions to: {PRED_CSV[:-4]+'.txt'}")
