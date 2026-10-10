"""Kaggle training and offline submission for UPAR 2027 Track 2.

AIMv2-3B experiment: frozen, generic pretrained AIMv2-3B/14 patch features are
pooled over the whole image and its upper and lower halves, and a 40-attribute
head is trained on the cached features. The checkpoint and the ranking distance
are selected by validation retrieval mADM.
"""

import csv
import importlib.util
import json
import os
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


DATA_ROOT = Path(os.environ.get("UPAR_DATA_ROOT", "/kaggle/working/upar_2027/data"))
OUTPUT_ROOT = Path("/kaggle/working" if Path("/kaggle/working").exists() else ".")
MODEL_ID = "aimv2_3b_patch14_448.apple_pt"  # Generic image-text pretraining; no PAR weights.
USE_PRETRAINED_WEIGHTS = True  # Kaggle training only; no runtime download.
IMAGE_SIZE = (448, 224)  # height, width; both must be multiples of the 14-pixel patch.
FEATURE_BATCH_SIZE = 16  # Reduce to 8 if the Kaggle GPU runs out of memory.
HEAD_BATCH_SIZE = 2048
HEAD_EPOCHS = 60
HEAD_LR = 1e-3
PATIENCE = 3  # Validation checks without improvement before stopping.
NUM_WORKERS = 2
SELECTION_QUERIES = 256
SEED = 42


def read_annotations(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        # The released train.csv omits the image-column name in its header.
        names = header if len(header) == 40 else header[1:]
        names = [name.removeprefix("# ").strip() for name in names]
        if len(names) != 40:
            raise ValueError(f"Expected 40 attribute names in {path}")
        paths, labels = [], []
        for line, row in enumerate(reader, 2):
            if len(row) != 41:
                raise ValueError(f"{path}:{line}: expected image path and 40 labels")
            paths.append(row[0])
            labels.append([int(value) for value in row[1:]])
    labels = np.asarray(labels, dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: labels must be binary")
    return names, paths, labels


def image_transform(mean, std):
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class ImageDataset(Dataset):
    def __init__(self, root, paths, transform):
        self.root, self.paths, self.transform = root, paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class PortableRmsNorm(nn.Module):
    """RMSNorm from basic operations, computed in float32.

    The submission runs in half precision, where squaring large activations can
    overflow, and the traced graph must not depend on a newer PyTorch operator.
    """

    def __init__(self, norm):
        super().__init__()
        self.weight, self.eps = norm.weight, norm.eps

    def forward(self, tokens):
        wide = tokens.float()
        scale = wide.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (wide * scale).type_as(tokens) * self.weight


def replace_rms_norms(module):
    for name, child in module.named_children():
        if type(child).__name__ == "RmsNorm":
            setattr(module, name, PortableRmsNorm(child))
        else:
            replace_rms_norms(child)


class RegionPooledEncoder(nn.Module):
    """Mean of all patch tokens, of the upper half, and of the lower half."""

    def __init__(self, trunk):
        super().__init__()
        if trunk.num_prefix_tokens:
            raise ValueError("Expected a trunk without class or register tokens")
        self.trunk = trunk
        self.rows, self.columns = trunk.patch_embed.grid_size

    def forward(self, pixels):
        tokens = self.trunk.forward_features(pixels)
        grid = tokens.reshape(tokens.shape[0], self.rows, self.columns, tokens.shape[-1])
        half = self.rows // 2
        return torch.cat([tokens.mean(1), grid[:, :half].mean((1, 2)), grid[:, half:].mean((1, 2))], dim=-1)


class AttributeHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, 40)

    def forward(self, features):
        return self.linear(self.norm(features))


class SubmissionModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, pixels):
        return self.head(self.encoder(pixels))


@torch.inference_mode()
def extract_features(encoder, paths, transform, cache_path, device):
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (cached.get("paths") == paths and cached.get("model_id") == MODEL_ID
                and tuple(cached.get("image_size", ())) == tuple(IMAGE_SIZE)):
            print(f"Reusing {cache_path}")
            return cached["features"]
    loader = DataLoader(ImageDataset(DATA_ROOT, paths, transform), batch_size=FEATURE_BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    dtype = next(encoder.parameters()).dtype
    pieces = []
    for batch_number, pixels in enumerate(loader, 1):
        features = encoder(pixels.to(device, dtype=dtype, non_blocking=True))
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite image features; half precision overflowed")
        pieces.append(features.cpu().half())
        if batch_number % 100 == 0:
            print(f"Extracted {min(batch_number * FEATURE_BATCH_SIZE, len(paths)):,}/{len(paths):,}", flush=True)
    features = torch.cat(pieces)
    torch.save({"features": features, "paths": paths, "model_id": MODEL_ID,
                "image_size": tuple(IMAGE_SIZE)}, cache_path)
    return features


@torch.inference_mode()
def predict_logits(head, features, device):
    head.eval()
    return np.concatenate([
        head(batch.float().to(device)).float().cpu().numpy()
        for batch in features.split(HEAD_BATCH_SIZE)
    ])


def distances_for(logits, queries, method):
    if method == "nll":
        return np.logaddexp(0, logits).sum(axis=1)[None, :] - queries @ logits.T
    if method == "l1":
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return probabilities.sum(axis=1)[None, :] + queries @ (1 - 2 * probabilities).T
    raise ValueError(method)


def retrieval_metrics(logits, labels, queries, method="nll"):
    """Challenge mADM, mAP, Rank-1/5/10, mINP for a validation gallery."""
    labels = labels.astype(np.float32)
    queries = queries.astype(np.float32)
    gallery_sum = labels.sum(axis=1)
    totals = np.zeros(6, dtype=np.float64)
    for start in range(0, len(queries), 16):
        query_batch = queries[start:start + 16]
        distances = distances_for(logits, query_batch, method)
        agreement = 40 - gallery_sum[None, :] - query_batch.sum(axis=1)[:, None] + 2 * query_batch @ labels.T
        for row in range(len(query_batch)):
            order = np.argsort(distances[row], kind="stable")
            ranks = np.flatnonzero(agreement[row, order] == 40) + 1
            if not len(ranks):
                raise ValueError("Validation query has no exact gallery match")
            mean_agreement = agreement[row].mean()
            normalized = np.maximum(0, (agreement[row, order] - mean_agreement) / (40 - mean_agreement))
            prefix = np.cumsum(normalized)
            totals += [np.mean(prefix[ranks - 1] / ranks),
                       np.mean(np.arange(1, len(ranks) + 1) / ranks),
                       ranks[0] <= 1, ranks[0] <= 5, ranks[0] <= 10,
                       len(ranks) / ranks[-1]]
    keys = ("mADM", "mAP", "Rank-1", "Rank-5", "Rank-10", "mINP")
    return dict(zip(keys, (totals / len(queries)).tolist()))


def fit_head(train_features, train_labels, val_features, val_labels, device):
    head = AttributeHead(train_features.shape[1]).to(device)
    loader = DataLoader(TensorDataset(train_features, torch.from_numpy(train_labels.astype(np.float32))),
                        batch_size=HEAD_BATCH_SIZE, shuffle=True, pin_memory=device.type == "cuda")
    queries = np.unique(val_labels, axis=0).astype(np.float32)
    rng = np.random.default_rng(SEED)
    selected = queries[rng.choice(len(queries), min(SELECTION_QUERIES, len(queries)), replace=False)]
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    checkpoint_path = OUTPUT_ROOT / "upar_aimv2_3b_head.pth"
    best_score, stale = -1.0, 0
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        running = 0.0
        for features, targets in loader:
            features = features.float().to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(features), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(features)
        if epoch == 1 or epoch % 5 == 0 or epoch == HEAD_EPOCHS:
            logits = predict_logits(head, val_features, device)
            metrics = retrieval_metrics(logits, val_labels, selected, "nll")
            print(f"Epoch {epoch:02d}: BCE={running / len(train_labels):.4f}; selection={metrics}", flush=True)
            if metrics["mADM"] > best_score:
                best_score, stale = metrics["mADM"], 0
                torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            "epoch": epoch, "selection_mADM": best_score}, checkpoint_path)
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"Stopping at epoch {epoch}: no improvement in {PATIENCE} checks")
                    break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head.load_state_dict(checkpoint["head"])
    print(f"Restored epoch {checkpoint['epoch']} with selection mADM={best_score:.4f}")
    logits = predict_logits(head, val_features, device)
    candidates = {name: retrieval_metrics(logits, val_labels, selected, name)
                  for name in ("nll", "l1")}
    score_method = max(candidates, key=lambda name: candidates[name]["mADM"])
    print("Selection scoring comparison:", candidates)
    print("Chosen score:", score_method)
    print("Full validation metrics:", retrieval_metrics(logits, val_labels, queries, score_method))
    return head.cpu().eval(), score_method


RUN_PY = '''"""Offline AIMv2-3B attribute retrieval for UPAR 2027 Track 2."""
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CONFIG = None
TRANSFORM = None
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
BATCH_SIZE = 16


def load_model():
    global MODEL, CONFIG, TRANSFORM
    with (HERE / "assets/config.json").open() as handle:
        CONFIG = json.load(handle)
    TRANSFORM = transforms.Compose([
        transforms.Resize(tuple(CONFIG["image_size"]), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["mean"], CONFIG["std"]),
    ])
    # The weights are stored in half precision; a CPU fallback widens them again.
    MODEL = torch.jit.load(str(HERE / "assets/model.pt"), map_location=DEVICE).eval()
    if DTYPE == torch.float32:
        MODEL.float()


def rank_gallery(sample):
    if MODEL is None:
        load_model()
    names = sample["attribute_names"]
    indices = [CONFIG["attribute_names"].index(name) for name in names]
    queries = np.asarray(sample["queries"], dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != len(indices):
        raise ValueError("Expected a Q x 40 query matrix")
    gallery = sample["gallery"]
    distances = np.empty((len(queries), len(gallery)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(gallery), BATCH_SIZE):
            batch = gallery[start:start + BATCH_SIZE]
            pixels = []
            for item in batch:
                with Image.open(item["image_path"]) as image:
                    pixels.append(TRANSFORM(image.convert("RGB")))
            pixels = torch.stack(pixels).to(DEVICE, dtype=DTYPE)
            logits = MODEL(pixels).float().cpu().numpy()[:, indices]
            if CONFIG["score_method"] == "nll":
                block = np.logaddexp(0, logits).sum(axis=1)[None, :] - queries @ logits.T
            else:
                probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                block = probs.sum(axis=1)[None, :] + queries @ (1 - 2 * probs).T
            distances[:, start:start + len(batch)] = block
            if (start // BATCH_SIZE + 1) % 100 == 0:
                print(f"Ranked {start + len(batch):,}/{len(gallery):,} images", flush=True)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite distances")
    return {"distances": distances}
'''

METADATA_YAML = '''# Code submissions are executed by the challenges ingestion program, which
# imports run.py and calls rank_gallery(). Nothing in this
# file is used for inference, it exists only so the platform never treats the
# archive as a result submission. Leave it as-is.
command: python3 -c "print('Submission package ready. The ingestion program imports run.py')"
'''


def package_submission(encoder, head, names, score_method, mean, std, transform, example_path, example_query):
    folder = OUTPUT_ROOT / "upar_aimv2_3b_submission"
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = SubmissionModel(encoder.float().cpu().eval(), head.cpu().eval())
    # Plain attention keeps the traced graph loadable by the runner's older PyTorch.
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    with Image.open(example_path) as image:
        example = transform(image.convert("RGB")).unsqueeze(0)
    print("Tracing on the CPU in float32; this takes several minutes", flush=True)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
        batch = example.repeat(2, 1, 1, 1)
        reference = model(batch)
        if not torch.allclose(reference, traced(batch), atol=1e-4, rtol=1e-4):
            raise ValueError("TorchScript export changed model predictions")
        traced.half()  # Halves the archive; the full-precision export would be about 11 GB.
        traced.save(str(assets / "model.pt"))
    del traced, model
    (assets / "config.json").write_text(json.dumps({
        "attribute_names": names, "score_method": score_method,
        "image_size": list(IMAGE_SIZE), "mean": list(mean), "std": list(std),
        "model_id": MODEL_ID,
    }, indent=2))
    (folder / "run.py").write_text(RUN_PY)
    (folder / "metadata.yaml").write_text(METADATA_YAML)
    archive_path = OUTPUT_ROOT / "upar_aimv2_3b_submission.zip"
    expected = {"run.py", "metadata.yaml", "assets/model.pt", "assets/config.json"}
    # Stored, not deflated: floating-point weights barely compress.
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(expected):
            archive.write(folder / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        print("ZIP contents:", archive.namelist())
        assert set(archive.namelist()) == expected
        with tempfile.TemporaryDirectory() as temp_dir:
            archive.extractall(temp_dir)
            spec = importlib.util.spec_from_file_location("aimv2_submission_check", Path(temp_dir) / "run.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            query = example_query.astype(np.float32)[None, :]
            result = module.rank_gallery({
                "attribute_names": names,
                "queries": query.astype(int).tolist(),
                "gallery": [{"index": 0, "image_path": str(example_path)}],
            })
            distances = np.asarray(result["distances"])
            assert distances.shape == (1, 1) and np.isfinite(distances).all()
            expected_distance = distances_for(reference[:1].numpy(), query, score_method)
            print(f"Packaged inference smoke test passed; distance {distances[0, 0]:.4f} "
                  f"against {expected_distance[0, 0]:.4f} in float32")
    print(f"Upload this ZIP to Codabench: {archive_path} ({archive_path.stat().st_size / 1e9:.1f} GB)")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    train_csv = DATA_ROOT / "annotations/train.csv"
    val_csv = DATA_ROOT / "annotations/val.csv"
    if not train_csv.is_file() or not val_csv.is_file():
        raise FileNotFoundError("Run the notebook's download cells first")
    if IMAGE_SIZE[0] % 14 or IMAGE_SIZE[1] % 14:
        raise ValueError("IMAGE_SIZE must be a multiple of the 14-pixel patch size")
    names, train_paths, train_labels = read_annotations(train_csv)
    val_names, val_paths, val_labels = read_annotations(val_csv)
    if names != val_names:
        raise ValueError("Train/validation attribute orders differ")
    for path in train_paths[:10] + val_paths[:10]:
        if not (DATA_ROOT / path).is_file():
            raise FileNotFoundError(DATA_ROOT / path)
    print(f"Train={len(train_paths):,}, validation={len(val_paths):,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No GPU found: feature extraction will run in float32 on the CPU and be very slow")
    # timm resamples the 448 x 448 position embedding to the pedestrian aspect ratio.
    trunk = timm.create_model(MODEL_ID, pretrained=USE_PRETRAINED_WEIGHTS, num_classes=0, img_size=IMAGE_SIZE)
    mean, std = trunk.pretrained_cfg["mean"], trunk.pretrained_cfg["std"]
    replace_rms_norms(trunk)
    encoder = RegionPooledEncoder(trunk).eval().requires_grad_(False)
    if device.type == "cuda":
        encoder.half()
    encoder.to(device)
    transform = image_transform(mean, std)
    train_features = extract_features(encoder, train_paths, transform, OUTPUT_ROOT / "upar_aimv2_3b_train_features.pt", device)
    val_features = extract_features(encoder, val_paths, transform, OUTPUT_ROOT / "upar_aimv2_3b_val_features.pt", device)
    head, score_method = fit_head(train_features, train_labels, val_features, val_labels, device)
    del train_features, val_features
    encoder.cpu()
    torch.cuda.empty_cache()
    package_submission(encoder, head, names, score_method, mean, std, transform,
                       DATA_ROOT / val_paths[0], val_labels[0])


if __name__ == "__main__":
    main()
