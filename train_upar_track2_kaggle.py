"""Kaggle training script for UPAR 2027 Track 2.

Reads challenge data from /kaggle/working/upar_2027/data by default.
This file is also the source for the training cells in the accompanying notebook.
"""

import csv
import importlib.util
import os
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DATA_ROOT = Path(os.environ.get("UPAR_DATA_ROOT", "/kaggle/working/upar_2027/data"))
OUTPUT_ROOT = Path("/kaggle/working" if Path("/kaggle/working").exists() else ".")
EPOCHS = 5
BATCH_SIZE = 128
NUM_WORKERS = 2
LEARNING_RATE = 3e-4
USE_IMAGENET_WEIGHTS = True  # Needs Internet once, or a cached torchvision checkpoint.
IMAGE_SIZE = (256, 128)  # height, width
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


def main():
    if not (DATA_ROOT / "annotations/train.csv").is_file():
        raise FileNotFoundError(
            f"Missing {DATA_ROOT / 'annotations/train.csv'}. Run the download notebook "
            "in this Kaggle session or attach its saved output and set UPAR_DATA_ROOT."
        )
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

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), normalize,
    ])
    train_loader = DataLoader(
        AttributeDataset(DATA_ROOT, train_paths, train_labels, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        AttributeDataset(DATA_ROOT, val_paths, val_labels, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT if USE_IMAGENET_WEIGHTS else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 40)
    model.to(device)
    positives = train_labels.sum(axis=0)
    negative_to_positive = (len(train_labels) - positives) / np.maximum(positives, 1)
    pos_weight = torch.tensor(np.clip(negative_to_positive, 1, 20), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    checkpoint_path = OUTPUT_ROOT / "upar_track2_best.pth"
    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.item() * len(images)

        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    loss = criterion(model(images), labels)
                val_total += loss.item() * len(images)
        train_loss = train_total / len(train_paths)
        val_loss = val_total / len(val_paths)
        print(f"Epoch {epoch}/{EPOCHS}: train BCE={train_loss:.4f}, val BCE={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "attribute_names": names,
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"Saved {checkpoint_path}")

    package_submission(checkpoint_path, DATA_ROOT / val_paths[0], names, val_labels[0])


RUN_PY = '''"""UPAR 2027 Track 2 inference entry point."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
MODEL = None
ATTRIBUTE_NAMES = None
TRANSFORM = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global MODEL, ATTRIBUTE_NAMES, TRANSFORM
    checkpoint = torch.load(HERE / "assets/upar_track2_best.pth", map_location="cpu", weights_only=True)
    ATTRIBUTE_NAMES = checkpoint["attribute_names"]
    size = tuple(checkpoint["image_size"])
    TRANSFORM = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 40)
    model.load_state_dict(checkpoint["state_dict"])
    MODEL = model.to(DEVICE).eval()


def rank_gallery(sample):
    if MODEL is None:
        load_model()
    order = [ATTRIBUTE_NAMES.index(name) for name in sample["attribute_names"]]
    gallery = sample["gallery"]
    queries = np.asarray(sample["queries"], dtype=np.float32)
    scores = np.empty((len(queries), len(gallery)), dtype=np.float32)
    batch_size = 128
    with torch.inference_mode():
        for start in range(0, len(gallery), batch_size):
            batch = gallery[start:start + batch_size]
            images = []
            for entry in batch:
                with Image.open(entry["image_path"]) as image:
                    images.append(TRANSFORM(image.convert("RGB")))
            logits = MODEL(torch.stack(images).to(DEVICE))
            probabilities = logits.sigmoid().float().cpu().numpy()[:, order]
            # L1 distance from each binary query to each predicted attribute vector.
            distances = probabilities.sum(axis=1)[None, :] + queries @ (1.0 - 2.0 * probabilities).T
            scores[:, start:start + len(batch)] = distances
    return {"distances": scores}
'''

METADATA_YAML = '''# Code submissions are executed by the challenges ingestion program, which
# imports run.py and calls rank_gallery(). Nothing in this
# file is used for inference, it exists only so the platform never treats the
# archive as a result submission. Leave it as-is.
command: python3 -c "print('Submission package ready. The ingestion program imports run.py')"
'''


def package_submission(checkpoint_path, sample_image, names, sample_query):
    folder = OUTPUT_ROOT / "upar_track2_submission"
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(RUN_PY)
    (folder / "metadata.yaml").write_text(METADATA_YAML)
    shutil.copy2(checkpoint_path, assets / checkpoint_path.name)
    archive_path = OUTPUT_ROOT / "upar_track2_submission.zip"
    expected = {"run.py", "metadata.yaml", "assets/upar_track2_best.pth"}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(expected):
            archive.write(folder / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        print("ZIP contents:", archive.namelist())
        assert set(archive.namelist()) == expected
        with tempfile.TemporaryDirectory() as temp_dir:
            archive.extractall(temp_dir)
            spec = importlib.util.spec_from_file_location("upar_submission_check", Path(temp_dir) / "run.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.rank_gallery({
                "attribute_names": names,
                "queries": [sample_query.astype(int).tolist()],
                "gallery": [{"index": 0, "image_path": str(sample_image)}],
            })
            distances = np.asarray(result["distances"])
            assert distances.shape == (1, 1) and np.isfinite(distances).all()
            print("Packaged inference smoke test passed")
    print("Checkpoint:", checkpoint_path)
    print("Upload this ZIP to Codabench:", archive_path)


if __name__ == "__main__":
    main()
