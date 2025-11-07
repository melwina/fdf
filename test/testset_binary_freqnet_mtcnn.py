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
from facenet_pytorch import MTCNN

# ---------------- Config ----------------
TEST_PATH = "testset_annotations.csv"
ZIP_PATH = "ai_face_competition_testset.zip"
# MODEL_NAME    = "2align_binary_face_classifier_xception_early_stopped.pth"
MODEL_NAME = "freqnet_binary_face_classifier_xception_early_stopped_intersection_weighted.pth"
RESULTS_DIR   = "results"
RESULTS_FILE  = os.path.join(RESULTS_DIR, f"{MODEL_NAME[:14]}_binary_results_testset.txt")
PRED_CSV      = os.path.join(RESULTS_DIR, f"{MODEL_NAME[:14]}_binary_predictions_testset.csv")

IMG_SIZE    = 299
BATCH_SIZE  = 256
NUM_WORKERS = 8
PIN_MEMORY  = torch.cuda.is_available()

VAL_SAMPLE_N = None

# Initialize face aligner
face_aligner = MTCNN(image_size=IMG_SIZE, margin=20, post_process=False, device="cpu")

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

        # Apply face alignment using MTCNN
        aligned = face_aligner(img)
        if aligned is None:
            with open(self._fail_log, "a") as logf:
                logf.write(json.dumps({
                    "error": "No face detected",
                    "zip_path": str(row.zip_path),
                    "member": member,
                    "image_path": str(image_path_for_output)
                }) + "\n")
            # Fallback to resized original image
            img = img.resize((IMG_SIZE, IMG_SIZE))
        else:
            # Convert back to PIL for transforms
            img = Image.fromarray(aligned.permute(1, 2, 0).byte().cpu().numpy())

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
# ---------- High-pass for images (HFRI) ----------
class FreqHighPass(nn.Module):
    def __init__(self, cutoff: float = 0.10):
        super().__init__()
        self.cutoff = cutoff
        self._cache = None  # (H,W,device,dtype) -> mask

    def _mask(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing='ij')
        cy, cx = (H - 1)/2.0, (W - 1)/2.0
        rr = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = rr.max()
        m = (rr >= self.cutoff * rmax).to(torch.float32)[None, None]  # keep mask in fp32
        self._cache = (key, m)
        return m

    def forward(self, x):
        # x: (B, C, H, W)
        dtype = x.dtype
        H, W = x.shape[-2], x.shape[-1]
        m = self._mask(H, W, x.device, torch.float32)  # fp32 mask

        # Run FFT in fp32 with autocast OFF
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            X = torch.fft.fft2(x32, dim=(-2, -1))
            Xs = torch.fft.fftshift(X, dim=(-2, -1))
            Xhp = torch.fft.ifftshift(Xs * m, dim=(-2, -1))
            x_hp = torch.fft.ifft2(Xhp, dim=(-2, -1)).real

            eps = 1e-6
            std = x_hp.std(dim=(-2, -1), keepdim=True).clamp_min(eps)  # (B,C,1,1)
            x_hp = x_hp / std
            x_hp = torch.nan_to_num(x_hp, 0.0, 0.0, 0.0)

        return x_hp.to(dtype)


# ---------- HFRF: high-pass on feature maps ----------
class FeatureHighPass(nn.Module):
    """Spatial high-pass on feature maps (B, C, H, W), residual form."""
    def __init__(self, cutoff: float = 0.10, alpha: float = 0.5):
        super().__init__()
        self.hpf = FreqHighPass(cutoff)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, feat):
        return feat + self.alpha * self.hpf(feat)


# ---------- FCL: Frequency Convolution Layer ----------
import math

class FrequencyConvLayer(nn.Module):
    def __init__(self, channels: int, ksize: int = 3, beta: float = 0.5):
        super().__init__()
        pad = ksize // 2
        self.amp_dw = nn.Conv2d(channels, channels, ksize, padding=pad, groups=channels, bias=False)
        self.amp_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.ph_dw  = nn.Conv2d(channels, channels, ksize, padding=pad, groups=channels, bias=False)
        self.ph_pw  = nn.Conv2d(channels, channels, 1, bias=False)
        self.beta   = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            X   = torch.fft.fft2(x32, dim=(-2, -1))
            A   = torch.abs(X).clamp_min(1e-8)   # amplitude
            P   = torch.angle(X)                 # phase in radians

            A_hat = self.amp_pw(self.amp_dw(A))
            P_hat = self.ph_pw(self.ph_dw(P))

            # Stabilizers
            A_hat = A_hat.clamp_min(0.0)
            # Keep phase bounded; polar uses radians naturally
            P_hat = torch.remainder(P_hat + math.pi, 2*math.pi) - math.pi

            # Recombine via polar (magnitude, angle) -> complex
            X_hat = torch.polar(A_hat, P_hat)    # complex64
            y32   = torch.fft.ifft2(X_hat, dim=(-2, -1)).real
            y32   = torch.nan_to_num(y32, 0.0, 0.0, 0.0)

            out32 = x32 + self.beta * y32

        return out32.to(dtype)


# ---------- Hybrid Face Classifier with toggles ----------
class FaceClassifier(nn.Module):
    def __init__(self,
                 num_classes=2,
                 use_hfri=True,
                 use_hfrf=True,
                 use_fcl=True,
                 cutoff=0.10,
                 dropout=0.5):
        super().__init__()
        self.use_hfri = use_hfri
        self.use_hfrf = use_hfrf
        self.use_fcl  = use_fcl

        if self.use_hfri:
            self.hpf_in = FreqHighPass(cutoff=cutoff)
            self.stem1x1 = nn.Conv2d(6, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                w = torch.zeros((3, 6, 1, 1))
                for i in range(3):
                    w[i, i, 0, 0] = 1.0  # identity on RGB; HP starts ignored
                self.stem1x1.weight.copy_(w)

        self.backbone = timm.create_model('xception', pretrained=True)
        num_features = self.backbone.fc.in_features

        # Post-feature frequency modules
        if self.use_hfrf or self.use_fcl:
            # We'll apply on the last convolutional feature map returned by forward_features
            # so we need its channel count. For xception in timm, it's num_features before pooling.
            self.hfrf = FeatureHighPass(cutoff=cutoff, alpha=0.5) if self.use_hfrf else nn.Identity()
            self.fcl  = FrequencyConvLayer(channels=num_features, ksize=3, beta=0.5) if self.use_fcl else nn.Identity()

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1)  # Single logit output for binary classification
        )

    def forward(self, x):
        # HFRI at input (concat HP + 1x1 mix back to 3ch)
        if self.use_hfri:
            x_hp = self.hpf_in(x)
            x = torch.cat([x, x_hp], dim=1)
            x = self.stem1x1(x)

        # Extract conv features
        feat = self.backbone.forward_features(x)  # (B, C, H, W)

        # HFRF on features (residual high-pass)
        if self.use_hfrf:
            feat = self.hfrf(feat)

        # FCL on features (amp/phase conv in frequency domain, residual)
        if self.use_fcl:
            feat = self.fcl(feat)

        # Head: pool + fc (unchanged)
        pooled = self.backbone.global_pool(feat)
        out = self.backbone.fc(pooled)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceClassifier(num_classes=2, use_hfri=True, use_hfrf=True, use_fcl=True).to(device)
state = torch.load(os.path.join("models", MODEL_NAME), map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Model loaded from: models/{MODEL_NAME}")
print(f"Using device: {device}")

# ---------------- Evaluate ----------------
all_preds = []
all_image_paths = []
from tqdm import tqdm

pbar = tqdm(total=len(eval_ds), desc="Evaluating", unit="img")
with torch.inference_mode():
    for batch in eval_loader:
        images, source_keys, img_paths = batch
        images = images.to(device, non_blocking=True)

        # Model outputs single logit, apply sigmoid to get probability
        logits = model(images).squeeze(1)  # (B, 1) -> (B,)
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        all_preds.extend(probs.detach().cpu().tolist())
        all_image_paths.extend(list(img_paths))

        pbar.update(len(images))
pbar.close()

# ---------------- Save per-image predictions CSV ----------------
os.makedirs(RESULTS_DIR, exist_ok=True)
pred_df = pd.DataFrame({
    "Image Path": all_image_paths,  # from manifest image_path if present, else member
    "pred": all_preds  # Probability of being fake (class 1)
})
pred_df.to_csv(PRED_CSV, index=False)
print(f"Saved per-image predictions to: {PRED_CSV}")
print(f"Predictions are probabilities (0-1) where values > 0.5 indicate 'fake'")
