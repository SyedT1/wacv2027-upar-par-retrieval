"""SigLIP 2 FixRes attribute retrieval for UPAR 2027 Track 2.

Run in the accompanying Kaggle notebook after its data download cells.
The exported submission uses only PyTorch, NumPy, and Pillow.
"""

import csv
import importlib.util
import json
import os
import random
import shutil
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


DATA_ROOT = Path(os.environ.get("UPAR_DATA_ROOT", "/kaggle/working/upar_2027/data"))
OUTPUT_ROOT = Path("/kaggle/working" if Path("/kaggle/working").exists() else ".")
MODEL_ID = "google/siglip2-base-patch16-384"
IMAGE_SIZE = 384
FEATURE_BATCH_SIZE = 8  # Lower this to 4 if the GPU runs out of memory.
HEAD_BATCH_SIZE = 2048
HEAD_EPOCHS = 25
HEAD_LR = 1e-3
NUM_WORKERS = 2
VAL_QUERIES_FOR_SELECTION = 512  # Fixed subset for affordable epoch selection.
SEED = 42


def read_annotations(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        names = header if len(header) == 40 else header[1:]
        names = [name.removeprefix("# ").strip() for name in names]
        if len(names) != 40:
            raise ValueError(f"Expected 40 attributes in {path}, got {len(names)}")
        paths, labels = [], []
        for line_number, row in enumerate(reader, 2):
            if len(row) != 41:
                raise ValueError(f"{path}:{line_number}: expected image path and 40 labels")
            paths.append(row[0])
            labels.append([int(v) for v in row[1:]])
    labels = np.asarray(labels, dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Nonbinary labels in {path}")
    return names, paths, labels


def letterbox(image, size=IMAGE_SIZE):
    """Keep the narrow pedestrian crop intact when using a square FixRes model."""
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


class ImageDataset(Dataset):
    def __init__(self, root, paths, transform):
        self.root, self.paths, self.transform = root, paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            return self.transform(letterbox(image))


def image_transform(processor):
    # Match the checkpoint's own normalization. The input is already 384 square.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(processor.image_mean, processor.image_std),
    ])


class RegionFeatures(nn.Module):
    """Global, upper, middle, and lower pooled SigLIP patch features."""

    def __init__(self, vision):
        super().__init__()
        self.vision = vision

    def forward(self, pixels):
        out = self.vision(pixel_values=pixels)
        patches = out.last_hidden_state  # SigLIP FixRes has no CLS token.
        pooled = out.pooler_output
        grid = patches.reshape(-1, 24, 24, patches.shape[-1])
        upper = grid[:, :8].mean(dim=(1, 2))
        middle = grid[:, 8:16].mean(dim=(1, 2))
        lower = grid[:, 16:].mean(dim=(1, 2))
        return torch.cat((pooled, upper, middle, lower), dim=-1)


class AttributeHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 40),
        )

    def forward(self, features):
        return self.net(features)


class SubmissionModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, pixels):
        return self.head(self.encoder(pixels))


@torch.inference_mode()
def extract_features(encoder, root, paths, transform, cache_path, device):
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.shape[0] == len(paths):
            print(f"Using cached features: {cache_path} {tuple(cached.shape)}")
            return cached
    loader = DataLoader(
        ImageDataset(root, paths, transform), batch_size=FEATURE_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
    )
    outputs = []
    encoder.eval()
    for batch_number, images in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            features = encoder(images)
        outputs.append(features.float().cpu().half())
        if batch_number % 500 == 0:
            print(f"Extracted {min(batch_number * FEATURE_BATCH_SIZE, len(paths)):,}/{len(paths):,}")
    features = torch.cat(outputs)
    torch.save(features, cache_path)
    return features


def validation_queries(labels):
    # Every distinct validation label vector is a retrieval query.
    queries = np.unique(labels, axis=0)
    return queries.astype(np.float32)


def retrieve_scores(logits, queries, method):
    """Returns distances, with smaller values ranked first."""
    if method == "nll":
        # Bernoulli negative log likelihood: sum softplus(logit) - q @ logit.
        base = np.logaddexp(0, logits).sum(axis=1)
        return base[None, :] - queries @ logits.T
    if method == "l1":
        probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return probs.sum(axis=1)[None, :] + queries @ (1 - 2 * probs).T
    raise ValueError(method)


def retrieval_metrics(logits, labels, queries, method="nll"):
    """Exact challenge mADM, mAP, Rank-1/5/10, and mINP on validation."""
    labels = labels.astype(np.float32)
    values = np.zeros(6, dtype=np.float64)
    gallery_sum = labels.sum(axis=1)
    for start in range(0, len(queries), 16):
        query_batch = queries[start:start + 16]
        distances = retrieve_scores(logits, query_batch, method)
        agreements = 40 - gallery_sum[None, :] - query_batch.sum(axis=1)[:, None] + 2 * query_batch @ labels.T
        for row in range(len(query_batch)):
            order = np.argsort(distances[row], kind="stable")
            matches = agreements[row, order] == 40
            positions = np.flatnonzero(matches) + 1
            if not len(positions):
                raise ValueError("Validation query has no exact gallery match")
            mean_match = agreements[row].mean()
            denom = 40 - mean_match
            ndom = np.maximum(0, (agreements[row, order] - mean_match) / denom)
            prefix = np.cumsum(ndom)
            values[0] += np.mean(prefix[positions - 1] / positions)
            values[1] += np.mean(np.arange(1, len(positions) + 1) / positions)
            values[2] += positions[0] <= 1
            values[3] += positions[0] <= 5
            values[4] += positions[0] <= 10
            values[5] += len(positions) / positions[-1]
    return dict(zip(("mADM", "mAP", "Rank-1", "Rank-5", "Rank-10", "mINP"),
                    (values / len(queries)).tolist()))


@torch.inference_mode()
def predict_logits(head, features, device):
    head.eval()
    pieces = []
    for batch in features.split(HEAD_BATCH_SIZE):
        pieces.append(head(batch.float().to(device)).float().cpu().numpy())
    return np.concatenate(pieces)


def fit_head(train_features, train_labels, val_features, val_labels, names, device):
    head = AttributeHead(train_features.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(train_features, torch.from_numpy(train_labels.astype(np.float32))),
        batch_size=HEAD_BATCH_SIZE, shuffle=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # Unweighted to preserve useful probabilities.
    queries = validation_queries(val_labels)
    rng = np.random.default_rng(SEED)
    selection = queries[rng.choice(len(queries), min(VAL_QUERIES_FOR_SELECTION, len(queries)), replace=False)]
    best = -1.0
    best_path = OUTPUT_ROOT / "upar_siglip2_head.pth"
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        total = 0.0
        for features, targets in loader:
            features = features.float().to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
        if epoch == 1 or epoch % 5 == 0 or epoch == HEAD_EPOCHS:
            logits = predict_logits(head, val_features, device)
            metrics = retrieval_metrics(logits, val_labels, selection)
            print(f"Epoch {epoch:02d} train BCE={total / len(train_labels):.4f} selection {metrics}")
            if metrics["mADM"] > best:
                best = metrics["mADM"]
                torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                            "names": names, "epoch": epoch}, best_path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    head.load_state_dict(checkpoint["head"])
    logits = predict_logits(head, val_features, device)
    # Select the ranking score using the public validation gallery only.
    candidates = {name: retrieval_metrics(logits, val_labels, selection, name)
                  for name in ("nll", "l1")}
    score_method = max(candidates, key=lambda name: candidates[name]["mADM"])
    print("Ranking candidates on fixed validation query subset:", candidates)
    print("Selected ranking:", score_method)
    print("Full validation metrics:", retrieval_metrics(logits, val_labels, queries, score_method))
    return head.cpu().eval(), score_method


RUN_PY = '''"""Offline UPAR 2027 Track 2 SigLIP 2 submission."""
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CONFIG = None
TRANSFORM = None

def letterbox(image, size=384):
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize((max(1, round(image.width * scale)),
                            max(1, round(image.height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas

def load_model():
    global MODEL, CONFIG, TRANSFORM
    CONFIG = json.loads((HERE / "assets/config.json").read_text())
    MODEL = torch.jit.load(str(HERE / "assets/model.pt"), map_location=DEVICE).eval()
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["image_mean"], CONFIG["image_std"]),
    ])

def rank_gallery(sample):
    if MODEL is None:
        load_model()
    names = sample["attribute_names"]
    order = [CONFIG["attribute_names"].index(name) for name in names]
    queries = np.asarray(sample["queries"], dtype=np.float32)
    gallery = sample["gallery"]
    distances = np.empty((len(queries), len(gallery)), dtype=np.float32)
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(gallery), batch_size):
            batch = gallery[start:start + batch_size]
            images = []
            for item in batch:
                with Image.open(item["image_path"]) as image:
                    images.append(TRANSFORM(letterbox(image)))
            pixels = torch.stack(images).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = MODEL(pixels).float().cpu().numpy()[:, order]
            if CONFIG["score_method"] == "nll":
                block = np.logaddexp(0, logits).sum(axis=1)[None, :] - queries @ logits.T
            else:
                probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                block = probs.sum(axis=1)[None, :] + queries @ (1 - 2 * probs).T
            distances[:, start:start + len(batch)] = block
            if (start // batch_size + 1) % 500 == 0:
                print(f"Ranked {start + len(batch):,}/{len(gallery):,} gallery images", flush=True)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite retrieval distances")
    return {"distances": distances}
'''


def export_submission(encoder, head, processor, names, score_method, sample_image, sample_query):
    folder = OUTPUT_ROOT / "upar_siglip2_submission"
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = SubmissionModel(encoder.cpu().eval(), head.cpu().eval())
    example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False, check_trace=False)
        # Check that tracing did not freeze the batch dimension or change values.
        probe = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        np.testing.assert_allclose(traced(probe).numpy(), model(probe).numpy(), rtol=1e-3, atol=1e-3)
    traced.save(str(assets / "model.pt"))
    config = {"attribute_names": names, "score_method": score_method,
              "image_mean": processor.image_mean, "image_std": processor.image_std}
    (assets / "config.json").write_text(json.dumps(config))
    (folder / "run.py").write_text(RUN_PY)
    (folder / "metadata.yaml").write_text(
        "# Code submissions are executed by the challenges ingestion program, which\n"
        "# imports run.py and calls rank_gallery(). Nothing in this\n"
        "# file is used for inference, it exists only so the platform never treats the\n"
        "# archive as a result submission. Leave it as-is.\n"
        "command: python3 -c \"print('Submission package ready. The ingestion program imports run.py')\"\n"
    )
    archive_path = OUTPUT_ROOT / "upar_siglip2_submission.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("run.py", "metadata.yaml", "assets/model.pt", "assets/config.json"):
            archive.write(folder / name, name)
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(directory)
        spec = importlib.util.spec_from_file_location("upar_siglip2_submission_check", Path(directory) / "run.py")
        submission = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submission)
        result = submission.rank_gallery({
            "attribute_names": names,
            "queries": [sample_query.astype(int).tolist()],
            "gallery": [{"index": 0, "image_path": str(sample_image)}],
        })
        assert result["distances"].shape == (1, 1)
        assert np.isfinite(result["distances"]).all()
    print("Submission ZIP and offline smoke test passed:", archive_path)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not (DATA_ROOT / "annotations/train.csv").is_file():
        raise FileNotFoundError("Download the UPAR challenge data in the notebook first")
    train_names, train_paths, train_labels = read_annotations(DATA_ROOT / "annotations/train.csv")
    val_names, val_paths, val_labels = read_annotations(DATA_ROOT / "annotations/val.csv")
    if train_names != val_names:
        raise ValueError("Train and validation attribute order differs")
    for path in train_paths[:10] + val_paths[:10]:
        if not (DATA_ROOT / path).is_file():
            raise FileNotFoundError(DATA_ROOT / path)
    print(f"Images: train={len(train_paths):,}, validation={len(val_paths):,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    pretrained = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    if pretrained.config.vision_config.image_size != IMAGE_SIZE:
        raise ValueError("Unexpected SigLIP 2 image size")
    encoder = RegionFeatures(pretrained.vision_model).to(device).eval()
    del pretrained
    transform = image_transform(processor)
    train_features = extract_features(encoder, DATA_ROOT, train_paths, transform,
                                      OUTPUT_ROOT / "upar_siglip2_train_features.pt", device)
    val_features = extract_features(encoder, DATA_ROOT, val_paths, transform,
                                    OUTPUT_ROOT / "upar_siglip2_val_features.pt", device)
    head, score_method = fit_head(train_features, train_labels, val_features, val_labels, train_names, device)
    del train_features, val_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    export_submission(encoder, head, processor, train_names, score_method,
                      DATA_ROOT / val_paths[0], val_labels[0])


if __name__ == "__main__":
    main()
