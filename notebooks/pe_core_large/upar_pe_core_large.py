"""Kaggle training and offline submission for UPAR 2027 Track 2.

PE-Core Large experiment: the vision tower of Meta's image-text pretrained
Perception Encoder Core L/14 is fine-tuned end to end on the 40 challenge
attributes. The checkpoint and the ranking distance are selected by
validation retrieval mADM.
"""

import csv
import importlib.util
import json
import math
import os
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.optim import create_optimizer_v2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


DATA_ROOT = Path(os.environ.get("UPAR_DATA_ROOT", "/kaggle/working/upar_2027/data"))
OUTPUT_ROOT = Path("/kaggle/working" if Path("/kaggle/working").exists() else ".")
MODEL_ID = "vit_pe_core_large_patch14_336.fb"  # Image-text pretraining; no PAR weights.
USE_PRETRAINED_WEIGHTS = True  # Kaggle training only; no runtime download.
IMAGE_SIZE = (336, 168)  # height, width; both must be multiples of the 14-pixel patch.
EPOCHS = 3
BATCH_SIZE = 32  # Reduce to 16 if the Kaggle GPU still runs out of memory.
GRAD_CHECKPOINTING = True  # Recomputes block activations in the backward pass; needed on a 16 GB GPU.
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 2
LEARNING_RATE = 1e-4  # Head learning rate; earlier blocks are scaled down by LAYER_DECAY.
LAYER_DECAY = 0.75
WEIGHT_DECAY = 0.05
DROP_PATH_RATE = 0.1
WARMUP_STEPS = 500
LOG_EVERY = 200
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
            raise ValueError(f"Expected 40 attributes in {path}, got {len(names)}")
        paths, labels = [], []
        for row_number, row in enumerate(reader, 2):
            if len(row) != 41:
                raise ValueError(f"{path}:{row_number}: expected image path and 40 labels")
            paths.append(row[0])
            labels.append([int(value) for value in row[1:]])
    labels = np.asarray(labels, dtype=np.float32)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path} contains labels other than 0 and 1")
    return names, paths, labels


class AttributeDataset(Dataset):
    def __init__(self, root, paths, labels, transform):
        self.root, self.paths, self.labels, self.transform = root, paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.from_numpy(self.labels[index])


def build_transforms(mean, std):
    resize = transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC)
    normalize = transforms.Normalize(mean, std)
    # No hue or saturation jitter: 24 of the 40 attributes are clothing colours.
    train_transform = transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMAGE_SIZE, padding=10, padding_mode="edge"),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    return train_transform, val_transform


def build_optimizer(model):
    optimizer = create_optimizer_v2(model, opt="adamw", lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY, layer_decay=LAYER_DECAY)
    # timm stores the layer-wise factor as lr_scale; PyTorch schedulers ignore it.
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * group.get("lr_scale", 1.0)
    return optimizer


@torch.inference_mode()
def predict_logits(model, loader, device):
    model.eval()
    pieces = []
    for images, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        pieces.append(logits.float().cpu().numpy())
    return np.concatenate(pieces)


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


RUN_PY = '''"""Offline PE-Core Large attribute retrieval for UPAR 2027 Track 2."""
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
BATCH_SIZE = 64


def load_model():
    global MODEL, CONFIG, TRANSFORM
    with (HERE / "assets/config.json").open() as handle:
        CONFIG = json.load(handle)
    TRANSFORM = transforms.Compose([
        transforms.Resize(tuple(CONFIG["image_size"]), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["mean"], CONFIG["std"]),
    ])
    MODEL = torch.jit.load(str(HERE / "assets/model.pt"), map_location=DEVICE).eval()


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
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                logits = MODEL(torch.stack(pixels).to(DEVICE)).float().cpu().numpy()[:, indices]
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


def package_submission(model, names, score_method, mean, std, val_transform, example_path, example_query):
    folder = OUTPUT_ROOT / "upar_pe_core_large_submission"
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    # The runner has no timm, so the model ships as TorchScript. Plain attention
    # keeps the traced graph loadable by the runner's older PyTorch.
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    with Image.open(example_path) as image:
        example = val_transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
        for batch in (example, example.repeat(2, 1, 1, 1)):
            if not torch.allclose(model(batch), traced(batch), atol=1e-4, rtol=1e-4):
                raise ValueError("TorchScript export changed model predictions")
        traced.save(str(assets / "model.pt"))
    (assets / "config.json").write_text(json.dumps({
        "attribute_names": names, "score_method": score_method,
        "image_size": list(IMAGE_SIZE), "mean": list(mean), "std": list(std),
        "model_id": MODEL_ID,
    }, indent=2))
    (folder / "run.py").write_text(RUN_PY)
    (folder / "metadata.yaml").write_text(METADATA_YAML)
    archive_path = OUTPUT_ROOT / "upar_pe_core_large_submission.zip"
    expected = {"run.py", "metadata.yaml", "assets/model.pt", "assets/config.json"}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(expected):
            archive.write(folder / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        print("ZIP contents:", archive.namelist())
        assert set(archive.namelist()) == expected
        with tempfile.TemporaryDirectory() as temp_dir:
            archive.extractall(temp_dir)
            spec = importlib.util.spec_from_file_location("pe_core_large_submission_check", Path(temp_dir) / "run.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.rank_gallery({
                "attribute_names": names,
                "queries": [example_query.astype(int).tolist()],
                "gallery": [{"index": 0, "image_path": str(example_path)}],
            })
            distances = np.asarray(result["distances"])
            assert distances.shape == (1, 1) and np.isfinite(distances).all()
            print("Packaged inference smoke test passed")
    print(f"Upload this ZIP to Codabench: {archive_path} ({archive_path.stat().st_size / 1e6:,.0f} MB)")


def main():
    if not (DATA_ROOT / "annotations/train.csv").is_file():
        raise FileNotFoundError(
            f"Missing {DATA_ROOT / 'annotations/train.csv'}. Run the download cells above "
            "in this Kaggle session or attach its saved output and set UPAR_DATA_ROOT."
        )
    if IMAGE_SIZE[0] % 14 or IMAGE_SIZE[1] % 14:
        raise ValueError("IMAGE_SIZE must be a multiple of the 14-pixel patch size")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    names, train_paths, train_labels = read_annotations(DATA_ROOT / "annotations/train.csv")
    val_names, val_paths, val_labels = read_annotations(DATA_ROOT / "annotations/val.csv")
    if names != val_names:
        raise ValueError("Train and validation attribute orders differ")
    for relative in train_paths[:10] + val_paths[:10]:
        if not (DATA_ROOT / relative).is_file():
            raise FileNotFoundError(f"Missing image {DATA_ROOT / relative}; finish downloading the images")
    print(f"Loaded {len(train_paths):,} train and {len(val_paths):,} validation images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # timm resamples the 336 x 336 position embedding to the pedestrian aspect ratio.
    model = timm.create_model(MODEL_ID, pretrained=USE_PRETRAINED_WEIGHTS, num_classes=40,
                              img_size=IMAGE_SIZE, drop_path_rate=DROP_PATH_RATE)
    mean, std = model.pretrained_cfg["mean"], model.pretrained_cfg["std"]
    model.set_grad_checkpointing(GRAD_CHECKPOINTING)
    model.to(device)

    train_transform, val_transform = build_transforms(mean, std)
    train_loader = DataLoader(
        AttributeDataset(DATA_ROOT, train_paths, train_labels, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        AttributeDataset(DATA_ROOT, val_paths, val_labels, val_transform),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = min(WARMUP_STEPS, total_steps // 10)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    queries = np.unique(val_labels, axis=0).astype(np.float32)
    rng = np.random.default_rng(SEED)
    selected = queries[rng.choice(len(queries), min(SELECTION_QUERIES, len(queries)), replace=False)]

    checkpoint_path = OUTPUT_ROOT / "upar_pe_core_large_best.pth"
    best_score, best_logits = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(images)
            seen += len(images)
            if step % LOG_EVERY == 0:
                print(f"Epoch {epoch} step {step:,}/{len(train_loader):,}: BCE={running / seen:.4f}", flush=True)

        logits = predict_logits(model, val_loader, device)
        metrics = retrieval_metrics(logits, val_labels, selected, "nll")
        print(f"Epoch {epoch}/{EPOCHS}: train BCE={running / seen:.4f}; selection={metrics}", flush=True)
        if metrics["mADM"] > best_score:
            best_score, best_logits = metrics["mADM"], logits
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "attribute_names": names,
                "image_size": IMAGE_SIZE,
                "model_id": MODEL_ID,
                "epoch": epoch,
                "selection_mADM": best_score,
            }, checkpoint_path)
            print(f"Saved {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Restored epoch {checkpoint['epoch']} with selection mADM={best_score:.4f}")
    candidates = {name: retrieval_metrics(best_logits, val_labels, selected, name)
                  for name in ("nll", "l1")}
    score_method = max(candidates, key=lambda name: candidates[name]["mADM"])
    print("Selection scoring comparison:", candidates)
    print("Chosen score:", score_method)
    print("Full validation metrics:", retrieval_metrics(best_logits, val_labels, queries, score_method))

    del optimizer, scaler
    model.set_grad_checkpointing(False)
    model.cpu()
    torch.cuda.empty_cache()
    package_submission(model, names, score_method, mean, std, val_transform,
                       DATA_ROOT / val_paths[0], val_labels[0])


if __name__ == "__main__":
    main()
