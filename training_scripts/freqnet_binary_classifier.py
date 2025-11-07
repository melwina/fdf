import os, io, json, random
import pandas as pd
from zipfile import ZipFile
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import timm

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from binary_test_fn import number_of_incorrect_predictions


# ---------------- Repro & Backend ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ---------------- Config ----------------
MANIFEST_PATH = "manifest.csv"
IMG_SIZE = 299                # Xception input size
BATCH_SIZE = 64
TRAIN_TARGET = 90000
VAL_TARGET   = 30000
TEST_TARGET  = 30000
NUM_WORKERS  = 4
PIN_MEMORY   = torch.cuda.is_available()
FAIL_LOG     = "loader_failures.log"
num_epochs = 40

# ---------------- Config Test ----------------
# MANIFEST_PATH = "manifest.csv"
# IMG_SIZE = 128          # instead of 299 → faster resize & less memory
# BATCH_SIZE = 8          # smaller batch size → lighter GPU load
# TRAIN_TARGET = 1000   # instead of 80_000
# VAL_TARGET   = 50      # instead of 20_000
# TEST_TARGET = 50
# NUM_WORKERS  = 2        # keep I/O simpler for testing
# PIN_MEMORY   = False    # can turn off for small runs
# FAIL_LOG     = "loader_failures.log"
# num_epochs = 4

class_names = ['real','fake']
num_classes = len(class_names)


# ---------------- Early Stopping ----------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-8, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f"New best validation loss: {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")
        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered! Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch + 1}")
            if self.restore_best_weights:
                self.restore_checkpoint(model)
                if self.verbose:
                    print("Restored best model weights")
            return True
        return False
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    
    def restore_checkpoint(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# ---------------- Device ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------- Transforms ----------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------------- Manifest-backed Dataset ----------------
def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")

class ManifestDataset(Dataset):
    classes = ['real', 'fake']  # 0=real, 1=fake

    def __init__(self, df: pd.DataFrame, transform=None, label_col="binary_label", split_col=None, split_value=None, failure_log_path=FAIL_LOG):
        if split_col is not None and split_value is not None:
            df = df[df[split_col] == split_value].copy()
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

        self._zip_handles = {}   # per-process cache
        self._pid = os.getpid()
        self._fail_log = failure_log_path

        required_cols = {"zip_path","member","Image Path", label_col}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {missing}")

    def __len__(self):
        return len(self.df)

    def _get_zip(self, zip_path: str) -> ZipFile:
        if os.getpid() != self._pid:
            self._zip_handles = {}
            self._pid = os.getpid()
        zf = self._zip_handles.get(zip_path)
        if zf is None:
            zf = ZipFile(zip_path, 'r')
            self._zip_handles[zip_path] = zf
        return zf

    def _read_image(self, row) -> Image.Image:
        zip_path  = str(row.zip_path) if pd.notna(row.zip_path) else ""
        member    = str(row.member)

        if not zip_path or zip_path.strip() == "" or not zip_path.lower().endswith(".zip"):
            return Image.open(_norm_path(member)).convert("RGB")

        zf = self._get_zip(_norm_path(zip_path))
        with zf.open(member) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row[self.label_col])

        try:
            img = self._read_image(row)
        except Exception as e:
            with open(self._fail_log, "a") as logf:
                logf.write(json.dumps({
                    "error": str(e),
                    "zip_path": str(row.zip_path),
                    "member": str(row.member),
                    "image_path": str(row["Image Path"]),
                }) + "\n")
            x = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            return x, label

        if self.transform:
            img = self.transform(img)
        return img, label

    def __del__(self):
        for z in list(self._zip_handles.values()):
            try: z.close()
            except: pass

# ---------------- Build datasets & loaders ----------------
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv(MANIFEST_PATH)
df = df.drop_duplicates(subset=["Image Path"])

# Pools from manifest
train_pool_ds = ManifestDataset(df, transform=train_transform, label_col="binary_label",
                                split_col="train", split_value=1)
test_pool_ds  = ManifestDataset(df, transform=val_transform,   label_col="binary_label",
                                split_col="val",   split_value=1)   # frozen pool; we'll sample test from here

print(f"Train pool size: {len(train_pool_ds)} | Test pool size: {len(test_pool_ds)}")


g = torch.Generator().manual_seed(SEED)

# ---- Disjoint train/new-val from the *train pool* (stratified) ----
pool_n = len(train_pool_ds)
desired_total = min(pool_n, TRAIN_TARGET + VAL_TARGET)
# pick a candidate subset of the train pool to split
perm = torch.randperm(pool_n, generator=g).numpy()
candidate_idx = perm[:desired_total]
labels_pool = train_pool_ds.df[train_pool_ds.label_col].astype(int).values
candidate_labels = labels_pool[candidate_idx]

# fraction for validation inside the candidate subset
val_fraction = (VAL_TARGET / max(1, (TRAIN_TARGET + VAL_TARGET)))
sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=SEED)
tr_sub, va_sub = next(sss.split(candidate_idx, candidate_labels))

train_indices = candidate_idx[tr_sub].tolist()
val_indices   = candidate_idx[va_sub].tolist()

# ---- Sampled test from the *old val* pool ----
def sample_indices(n_total, n_target, generator):
    n = min(n_total, n_target)
    perm = torch.randperm(n_total, generator=generator).tolist()
    return perm[:n]

test_indices = sample_indices(len(test_pool_ds), TEST_TARGET, g)

# ---- Samplers & Loaders ----
train_sampler = SubsetRandomSampler(train_indices, generator=g)
val_sampler   = SubsetRandomSampler(val_indices,   generator=g)
test_sampler  = SubsetRandomSampler(test_indices,  generator=g)

train_loader = DataLoader(
    train_pool_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
    prefetch_factor=2,
)

val_loader = DataLoader(
    train_pool_ds,   # same dataset; different sampler
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
    prefetch_factor=2,
)

test_loader = DataLoader(
    test_pool_ds,
    batch_size=BATCH_SIZE,
    sampler=test_sampler,   # sampled test, as requested
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
    prefetch_factor=2,
)

print(f"Using {len(train_indices)} train | {len(val_indices)} val (from train pool) | {len(test_indices)} test (from val pool)")
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
            nn.Linear(num_features, num_classes)
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

model = FaceClassifier(num_classes=2, use_hfri=True, use_hfrf=True, use_fcl=True).to(device)

# ---------------- Loss/Optim/Sched + AMP ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

early_stopping = EarlyStopping(patience=10, min_delta=1e-8, restore_best_weights=True, verbose=True)

# ---------------- Train / Validate ----------------
# Asserts that a tensor is finite (no NaNs or Infs)
# Added gradient clipping and scaler.unscale

def _assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        mx = torch.nanmax(tensor).item()
        mn = torch.nanmin(tensor).item()
        raise RuntimeError(f"[NonFinite] {name} min={mn:.6g} max={mx:.6g}")

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(data)
            _assert_finite(outputs, f"train_outputs@b{batch_idx}")   # <-- add
            loss = criterion(outputs, target)
            _assert_finite(loss,    f"train_loss@b{batch_idx}")      # <-- add
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                   # <-- add
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # <-- add (optional)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    return running_loss / len(loader), 100. * correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(data)
                _assert_finite(outputs, f"eval_outputs@b{batch_idx}")  # <-- add
                loss = criterion(outputs, target)
                _assert_finite(loss,    f"eval_loss@b{batch_idx}") 

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_preds.extend(predicted.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
    return running_loss / len(loader), 100. * correct / total, all_preds, all_targets

# ---------------- Training Loop ----------------

train_losses, train_accs = [], []
val_losses, val_accs = [], []

print("Starting training...")
print(f"Early stopping: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
print("-" * 70)

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc, preds, targets = validate_epoch(model, val_loader, criterion, device)

    train_losses.append(train_loss); train_accs.append(train_acc)
    val_losses.append(val_loss);     val_accs.append(val_acc)

    scheduler.step()

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%')

    if epoch % 5 == 0 and epoch != 0:
        number_of_incorrect_predictions(model, epoch, "intersection_weighted")

    if early_stopping(val_loss, model, epoch):
        print(f"\nTraining stopped early at epoch {epoch + 1}")
        break

# ---------------- Final Evaluation ----------------
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Training completed after {len(val_losses)} epochs")
print(f"Best Validation Loss (internal): {early_stopping.best_loss:.6f} at epoch {early_stopping.best_epoch + 1}")
print(f"Best Validation Accuracy (internal): {max(val_accs):.2f}%")

# Evaluate once on the frozen test set
test_loss, test_acc, test_preds, test_targets = validate_epoch(model, test_loader, criterion, device)
print(f"\nTEST Loss: {test_loss:.4f} | TEST Acc: {test_acc:.2f}%")

class_names = ['real','fake']
print("\nTEST Classification Report:")
print(classification_report(test_targets, test_preds, target_names=class_names))

cm = confusion_matrix(test_targets, test_preds)
print("\nTEST Confusion Matrix:")
print(cm)


# ---------------- Plots ----------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
plt.plot(val_losses,   label='Val Loss',   marker='s', markersize=3)
plt.axvline(x=early_stopping.best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({early_stopping.best_epoch + 1})')
plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Train Acc', marker='o', markersize=3)
plt.plot(val_accs,   label='Val Acc',   marker='s', markersize=3)
plt.axvline(x=early_stopping.best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({early_stopping.best_epoch + 1})')
plt.title('Training and Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(val_losses, label='Validation Loss', marker='s', markersize=4, color='orange')
plt.axhline(y=early_stopping.best_loss, color='red', linestyle='-', alpha=0.7, label=f'Best Val Loss ({early_stopping.best_loss:.6f})')
plt.axvline(x=early_stopping.best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({early_stopping.best_epoch + 1})')
plt.title('Validation Loss with Early Stopping'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('./plots', exist_ok=True)
plt.savefig('./plots/freqnet_binary_training_curves_intersection_weighted.png', dpi=300, bbox_inches='tight')
print("Training curves saved as './plots/freqnet_binary_training_curves_intersection_weighted.png'")

# ---------------- Save Model ----------------
model_path = os.path.join(os.getcwd(), 'models', 'freqnet_binary_face_classifier_xception_early_stopped_intersection_weighted.pth')
torch.save(model.state_dict(), model_path)
print(f"\nModel saved as '{model_path}'")
print("Note: Saved model contains the best weights from the training process")
