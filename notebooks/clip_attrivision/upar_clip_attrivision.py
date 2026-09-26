"""Kaggle training and offline submission for UPAR 2027 Track 2.

AttriVision-inspired CLIP experiment: generic CLIP image/text features,
positive/negative attribute prompts, and a focal-loss attribute head.
This is an adaptation for the challenge's 40-label retrieval metric, not a
reproduction of the AttriVision paper's training protocol or reported scores.
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
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizerFast


DATA_ROOT = Path(os.environ.get("UPAR_DATA_ROOT", "/kaggle/working/upar_2027/data"))
OUTPUT_ROOT = Path("/kaggle/working" if Path("/kaggle/working").exists() else ".")
MODEL_ID = "openai/clip-vit-base-patch32"  # Generic pretraining; no PAR weights.
IMAGE_SIZE = 224
FEATURE_BATCH_SIZE = 128  # Reduce to 64 if the Kaggle GPU runs out of memory.
HEAD_BATCH_SIZE = 2048
HEAD_EPOCHS = 100
HEAD_LR = 1e-3
NUM_WORKERS = 2
SELECTION_QUERIES = 256
SEED = 42

POSITIVE_PROMPTS = [
    "a photo of a young pedestrian", "a photo of an adult pedestrian",
    "a photo of an elderly pedestrian", "a photo of a female pedestrian",
    "a pedestrian with short hair", "a pedestrian with long hair",
    "a bald pedestrian", "a pedestrian wearing a short-sleeved top",
    "a pedestrian wearing a black top", "a pedestrian wearing a blue top",
    "a pedestrian wearing a brown top", "a pedestrian wearing a green top",
    "a pedestrian wearing a grey top", "a pedestrian wearing an orange top",
    "a pedestrian wearing a pink top", "a pedestrian wearing a purple top",
    "a pedestrian wearing a red top", "a pedestrian wearing a white top",
    "a pedestrian wearing a yellow top", "a pedestrian wearing a top of another color",
    "a pedestrian wearing shorts or a short skirt",
    "a pedestrian wearing black bottoms", "a pedestrian wearing blue bottoms",
    "a pedestrian wearing brown bottoms", "a pedestrian wearing green bottoms",
    "a pedestrian wearing grey bottoms", "a pedestrian wearing orange bottoms",
    "a pedestrian wearing pink bottoms", "a pedestrian wearing purple bottoms",
    "a pedestrian wearing red bottoms", "a pedestrian wearing white bottoms",
    "a pedestrian wearing yellow bottoms", "a pedestrian wearing bottoms of another color",
    "a pedestrian wearing trousers or shorts", "a pedestrian wearing a skirt or dress",
    "a pedestrian carrying a backpack", "a pedestrian carrying a bag",
    "a pedestrian wearing prescription glasses", "a pedestrian wearing sunglasses",
    "a pedestrian wearing a hat",
]
NEGATIVE_PROMPTS = [
    "a photo of a pedestrian who is not young", "a photo of a pedestrian who is not an adult",
    "a photo of a pedestrian who is not elderly", "a photo of a male pedestrian",
    "a pedestrian without short hair", "a pedestrian without long hair",
    "a pedestrian who is not bald", "a pedestrian wearing a long-sleeved top",
    *[f"a pedestrian not wearing a {color} top" for color in
      ("black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow")],
    "a pedestrian wearing a top with a named color",
    "a pedestrian wearing long pants or a long skirt",
    *[f"a pedestrian not wearing {color} bottoms" for color in
      ("black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow")],
    "a pedestrian wearing bottoms with a named color",
    "a pedestrian not wearing trousers or shorts", "a pedestrian not wearing a skirt or dress",
    "a pedestrian without a backpack", "a pedestrian without a bag",
    "a pedestrian without prescription glasses", "a pedestrian without sunglasses",
    "a pedestrian without a hat",
]


def read_annotations(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
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


def image_transform():
    return transforms.Compose([
        transforms.Lambda(lambda image: ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


class ImageDataset(Dataset):
    def __init__(self, root, paths):
        self.root, self.paths, self.transform = root, paths, image_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class CLIPImageEncoder(nn.Module):
    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixels):
        pooled = self.vision_model(pixel_values=pixels, return_dict=False)[1]
        return F.normalize(self.visual_projection(pooled), dim=-1)


class AttributeHead(nn.Module):
    def __init__(self, prompt_delta):
        super().__init__()
        if prompt_delta.shape[0] != 40:
            raise ValueError("Expected one prompt difference per attribute")
        self.register_buffer("prompt_delta", prompt_delta.float())
        self.residual = nn.Linear(prompt_delta.shape[1], 40)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, features):
        return self.log_scale.exp() * (features @ self.prompt_delta.T) + self.residual(features)


class SubmissionModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, pixels):
        return self.head(self.encoder(pixels))


def make_prompt_delta(clip_model, device):
    if len(POSITIVE_PROMPTS) != 40 or len(NEGATIVE_PROMPTS) != 40:
        raise ValueError("Prompt list must match the challenge's 40 attributes")
    tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    tokens = tokenizer(POSITIVE_PROMPTS + NEGATIVE_PROMPTS, padding=True,
                       truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        text_features = clip_model.get_text_features(**tokens).float()
        text_features = F.normalize(text_features, dim=-1)
    return (text_features[:40] - text_features[40:]).cpu()


@torch.inference_mode()
def extract_features(encoder, paths, cache_path, device):
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("paths") == paths and cached.get("model_id") == MODEL_ID:
            print(f"Reusing {cache_path}")
            return cached["features"]
    loader = DataLoader(ImageDataset(DATA_ROOT, paths), batch_size=FEATURE_BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    pieces = []
    encoder.eval()
    for batch_number, pixels in enumerate(loader, 1):
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            features = encoder(pixels.to(device, non_blocking=True))
        pieces.append(features.float().cpu().half())
        if batch_number % 100 == 0:
            print(f"Extracted {min(batch_number * FEATURE_BATCH_SIZE, len(paths)):,}/{len(paths):,}", flush=True)
    features = torch.cat(pieces)
    torch.save({"features": features, "paths": paths, "model_id": MODEL_ID}, cache_path)
    return features


def focal_bce(logits, targets, gamma=2.0):
    ordinary = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_correct = torch.exp(-ordinary)
    return ((1 - p_correct).pow(gamma) * ordinary).mean()


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


def fit_head(train_features, train_labels, val_features, val_labels, prompt_delta, device):
    head = AttributeHead(prompt_delta).to(device)
    loader = DataLoader(TensorDataset(train_features, torch.from_numpy(train_labels.astype(np.float32))),
                        batch_size=HEAD_BATCH_SIZE, shuffle=True, pin_memory=True)
    queries = np.unique(val_labels, axis=0).astype(np.float32)
    rng = np.random.default_rng(SEED)
    selected = queries[rng.choice(len(queries), min(SELECTION_QUERIES, len(queries)), replace=False)]
    best_score = -1.0
    start_epoch = 1
    checkpoint_path = OUTPUT_ROOT / "upar_clip_attrivision_head.pth"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        head.load_state_dict(checkpoint["head"])
        best_score = float(checkpoint["selection_mADM"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming the best head from epoch {start_epoch - 1}; selection mADM={best_score:.4f}")
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    if start_epoch > 1 and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    for epoch in range(start_epoch, HEAD_EPOCHS + 1):
        head.train()
        running = 0.0
        for features, targets in loader:
            features = features.float().to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = focal_bce(head(features), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(features)
        if epoch == 1 or epoch % 5 == 0 or epoch == HEAD_EPOCHS:
            logits = predict_logits(head, val_features, device)
            metrics = retrieval_metrics(logits, val_labels, selected, "nll")
            print(f"Epoch {epoch:02d}: focal loss={running / len(train_labels):.4f}; selection={metrics}")
            if metrics["mADM"] > best_score:
                best_score = metrics["mADM"]
                torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            "epoch": epoch, "selection_mADM": best_score,
                            "optimizer": optimizer.state_dict()}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head.load_state_dict(checkpoint["head"])
    logits = predict_logits(head, val_features, device)
    candidates = {name: retrieval_metrics(logits, val_labels, selected, name)
                  for name in ("nll", "l1")}
    score_method = max(candidates, key=lambda name: candidates[name]["mADM"])
    print("Selection scoring comparison:", candidates)
    print("Chosen score:", score_method)
    print("Full validation metrics:", retrieval_metrics(logits, val_labels, queries, score_method))
    return head.cpu().eval(), score_method


RUN_PY = '''"""Offline CLIP attribute retrieval for UPAR 2027 Track 2."""
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CONFIG = None

def load_model():
    global MODEL, CONFIG
    with (HERE / "assets/config.json").open() as handle:
        CONFIG = json.load(handle)
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
    transform = transforms.Compose([
        transforms.Lambda(lambda image: ImageOps.pad(image, (224, 224), color=(0, 0, 0))),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    with torch.inference_mode():
        for start in range(0, len(gallery), 128):
            batch = gallery[start:start + 128]
            pixels = []
            for item in batch:
                with Image.open(item["image_path"]) as image:
                    pixels.append(transform(image.convert("RGB")))
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                logits = MODEL(torch.stack(pixels).to(DEVICE)).float().cpu().numpy()[:, indices]
            if CONFIG["score_method"] == "nll":
                block = np.logaddexp(0, logits).sum(axis=1)[None, :] - queries @ logits.T
            else:
                probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                block = probs.sum(axis=1)[None, :] + queries @ (1 - 2 * probs).T
            distances[:, start:start + len(batch)] = block
            if (start // 128 + 1) % 100 == 0:
                print(f"Ranked {start + len(batch):,}/{len(gallery):,} images", flush=True)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite distances")
    return {"distances": distances}
'''


def package_submission(encoder, head, names, score_method, example_path, example_query):
    folder = OUTPUT_ROOT / "upar_clip_attrivision_submission"
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = SubmissionModel(encoder.cpu().eval(), head.cpu().eval())
    example = image_transform()(Image.open(example_path).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
        for batch in (example, example.repeat(2, 1, 1, 1)):
            if not torch.allclose(model(batch), traced(batch), atol=1e-4, rtol=1e-4):
                raise ValueError("TorchScript export changed model predictions")
        traced.save(str(assets / "model.pt"))
    (assets / "config.json").write_text(
        __import__("json").dumps({"attribute_names": names, "score_method": score_method}, indent=2)
    )
    (folder / "run.py").write_text(RUN_PY)
    metadata = Path("/kaggle/working/UPAR-Challenge-2027/examples/task2/sample_code_submission/metadata.yaml")
    if not metadata.is_file():
        metadata = Path(__file__).resolve().parents[2] / "UPAR-Challenge-2027/examples/task2/sample_code_submission/metadata.yaml"
    shutil.copy2(metadata, folder / "metadata.yaml")
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(OUTPUT_ROOT / "upar_clip_attrivision_submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in folder.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(folder))
        with zipfile.ZipFile(OUTPUT_ROOT / "upar_clip_attrivision_submission.zip") as archive:
            archive.extractall(directory)
        spec = importlib.util.spec_from_file_location("clip_submission_check", Path(directory) / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.rank_gallery({
            "attribute_names": names,
            "queries": [example_query.tolist()],
            "gallery": [{"index": 0, "image_path": str(example_path)}],
        })
        assert np.asarray(result["distances"]).shape == (1, 1)
        assert np.isfinite(result["distances"]).all()
    print("Submission ZIP:", OUTPUT_ROOT / "upar_clip_attrivision_submission.zip")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    train_csv = DATA_ROOT / "annotations/train.csv"
    val_csv = DATA_ROOT / "annotations/val.csv"
    if not train_csv.is_file() or not val_csv.is_file():
        raise FileNotFoundError("Run the notebook's download cells first")
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
        raise RuntimeError("Enable a Kaggle GPU for CLIP feature extraction")
    clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    prompt_delta = make_prompt_delta(clip_model, device)
    encoder = CLIPImageEncoder(clip_model.vision_model, clip_model.visual_projection).eval()
    del clip_model
    train_features = extract_features(encoder, train_paths, OUTPUT_ROOT / "upar_clip_train_features.pt", device)
    val_features = extract_features(encoder, val_paths, OUTPUT_ROOT / "upar_clip_val_features.pt", device)
    head, score_method = fit_head(train_features, train_labels, val_features, val_labels, prompt_delta, device)
    del train_features, val_features
    encoder.cpu()
    torch.cuda.empty_cache()
    package_submission(encoder, head, names, score_method, DATA_ROOT / val_paths[0], val_labels[0])


if __name__ == "__main__":
    main()
