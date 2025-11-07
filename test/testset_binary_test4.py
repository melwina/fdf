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

# ---------------- Config ----------------
TEST_PATH = "testset_annotations.csv"
ZIP_PATH = "ai_face_competition_testset.zip"
# MODEL_NAME    = "2align_binary_face_classifier_xception_early_stopped.pth"
MODEL_NAME = "messed_up_binary_face_classifier_xception_early_stopped_10.pth"
RESULTS_DIR   = "results"
RESULTS_FILE  = os.path.join(RESULTS_DIR, f"{MODEL_NAME[:10]}_binary_results_testset.txt")
PRED_CSV      = os.path.join(RESULTS_DIR, f"{MODEL_NAME[:10]}_binary_predictions_testset.csv")

IMG_SIZE    = 299
BATCH_SIZE  = 100
NUM_WORKERS = 8
PIN_MEMORY  = torch.cuda.is_available()

VAL_SAMPLE_N = None

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

class_names = ['real','fake']
num_classes = len(class_names)

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
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self._zip_handles = {}
        self._pid = os.getpid()
        self._fail_log = failure_log_path

        # Expect: "Image Path" == "{zipfilename.zip}/{member_path}"
        if "Image Path" not in self.df.columns:
            raise ValueError("manifest missing column: {'Image Path'}")

        zips, members = [], []
        for p in self.df["Image Path"].astype(str).fillna(""):
            p = p.lstrip("/").replace("\\", "/")
            if "/" in p:
                zip_seg, inner = p.split("/", 1)
                # Ensure it ends with .zip
                if not zip_seg.lower().endswith(".zip"):
                    zip_seg = f"{zip_seg}.zip"
                zips.append(zip_seg)
                members.append(inner)
            else:
                # No zip segment present; treat as direct filesystem path
                zips.append("")
                members.append(p)

        self.df["zip_path"] = zips
        self.df["member"]   = members

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

        if zip_path and zip_path.lower().endswith(".zip") and os.path.isfile(zip_path):
            zf = self._get_zip(_norm_path(zip_path))
            try:
                with zf.open(member) as f:
                    return Image.open(io.BytesIO(f.read())).convert("RGB")
            except KeyError:
                # Try to resolve when the CSV member omits internal folder(s)
                # e.g., CSV: "0000097.png" but zip has "ai_face_competition_testset/0000097.png"
                nm = zf.namelist()
                # Prefer exact basename suffix match and avoid directories
                candidates = [n for n in nm if not n.endswith("/") and (n.endswith("/" + member) or n == member)]
                if len(candidates) == 1:
                    resolved = candidates[0]
                    with zf.open(resolved) as f:
                        return Image.open(io.BytesIO(f.read())).convert("RGB")
                # If still ambiguous, try a stricter basename-only unique match
                base = os.path.basename(member)
                candidates = [n for n in nm if not n.endswith("/") and os.path.basename(n) == base]
                if len(candidates) == 1:
                    resolved = candidates[0]
                    with zf.open(resolved) as f:
                        return Image.open(io.BytesIO(f.read())).convert("RGB")
                # Could not resolve uniquely; re-raise to hit the fallback/log
                raise

        # Fallback: read from filesystem if not in a zip
        return Image.open(_norm_path(member)).convert("RGB")


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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
                logf.write(json.dumps({
                    "error": str(e),
                    "zip_path": str(row.zip_path),
                    "member": member,
                    "image_path": str(image_path_for_output)
                }) + "\n")
            x = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            return x, source_key, image_path_for_output

        if self.transform:
            img = self.transform(img)
        return img, source_key, image_path_for_output

    def __del__(self):
        for z in list(self._zip_handles.values()):
            try: z.close()
            except: pass

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
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x): return self.backbone(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceClassifier(num_classes=num_classes).to(device)
state = torch.load(os.path.join("models", MODEL_NAME), map_location=device)
model.load_state_dict(state)
model.eval()

# ---------------- Evaluate ----------------
all_preds = []
all_image_paths = []
from tqdm import tqdm

pbar = tqdm(total=len(eval_ds), desc="Evaluating", unit="img")
with torch.inference_mode():
    for batch in eval_loader:
        images, source_keys, img_paths = batch
        images = images.to(device, non_blocking=True)

        preds =  model(images)
        # preds = torch.softmax(model(images), dim=1)[:, 1]
        # preds = model(images).argmax(1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_image_paths.extend(list(img_paths))

        pbar.update(len(images))
pbar.close()

# ---------------- Save per-image predictions CSV ----------------
os.makedirs(RESULTS_DIR, exist_ok=True)
pred_df = pd.DataFrame({
    "Image Path": all_image_paths,  # from manifest image_path if present, else member
    "pred": all_preds
})
pred_df.to_csv(PRED_CSV, index=False)
print(f"Saved per-image predictions to: {PRED_CSV}")
