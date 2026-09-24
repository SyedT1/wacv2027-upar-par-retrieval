# UPAR Challenge 2027 — Track 2: Attribute-based Person Retrieval

This repository is for the UPAR Challenge Track 2 at the Real-World Surveillance workshop (WACV 2027). Given a query describing a person with 40 binary attributes, a submission ranks every pedestrian image in a gallery from best match to worst. The evaluation gallery comes from a hidden domain, so methods must generalize beyond the training datasets.

This README summarizes the supplied `upar-challenge-m.txt` challenge description. Consult the Codabench challenge page for current announcements and rules; the dates and requirements below reflect that captured description.

## Data and training rules

- A query contains all 40 attributes in the fixed order below. A gallery image is an exact match only when all 40 values agree. Multiple images may match one query.
- The hidden test data is mounted read-only under `input_data/`. The ingestion program passes parsed queries and gallery entries to the submission, so submissions do not need to read the CSV files directly.
- Fine-tuning pedestrian attribute recognition (PAR) on external data is prohibited. General-purpose pretrained models, such as ImageNet models or foundation models not specifically trained for PAR, are permitted. Do not use test images for training, pseudo-labeling, manual annotation, or hyperparameter selection.

| Index | Attribute | Index | Attribute |
| ---: | --- | ---: | --- |
| 0 | Age-Young | 20 | LowerBody-Length-Short |
| 1 | Age-Adult | 21 | LowerBody-Color-Black |
| 2 | Age-Old | 22 | LowerBody-Color-Blue |
| 3 | Gender-Female | 23 | LowerBody-Color-Brown |
| 4 | Hair-Length-Short | 24 | LowerBody-Color-Green |
| 5 | Hair-Length-Long | 25 | LowerBody-Color-Grey |
| 6 | Hair-Length-Bald | 26 | LowerBody-Color-Orange |
| 7 | UpperBody-Length-Short | 27 | LowerBody-Color-Pink |
| 8 | UpperBody-Color-Black | 28 | LowerBody-Color-Purple |
| 9 | UpperBody-Color-Blue | 29 | LowerBody-Color-Red |
| 10 | UpperBody-Color-Brown | 30 | LowerBody-Color-White |
| 11 | UpperBody-Color-Green | 31 | LowerBody-Color-Yellow |
| 12 | UpperBody-Color-Grey | 32 | LowerBody-Color-Other |
| 13 | UpperBody-Color-Orange | 33 | LowerBody-Type-Trousers&Shorts |
| 14 | UpperBody-Color-Pink | 34 | LowerBody-Type-Skirt&Dress |
| 15 | UpperBody-Color-Purple | 35 | Accessory-Backpack |
| 16 | UpperBody-Color-Red | 36 | Accessory-Bag |
| 17 | UpperBody-Color-White | 37 | Accessory-Glasses-Normal |
| 18 | UpperBody-Color-Yellow | 38 | Accessory-Glasses-Sun |
| 19 | UpperBody-Color-Other | 39 | Accessory-Hat |

## Submission API

Upload a ZIP of the **contents** of a submission folder, with `run.py` and the example submission's unchanged `metadata.yaml` at the archive root. Include model weights and any dependencies your code needs. The evaluation container has no network access and uses `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (Python 3.11, PyTorch 2.4.1, torchvision, NumPy, and Pillow).

The ingestion program imports `run.py`, optionally calls `load_model()` once, then calls `rank_gallery(sample)` once. The sample contains:

```python
{
    "attribute_names": ["Age-Young", "Age-Adult", ...],  # 40 names
    "queries": [[0, 1, 0, ...], ...],                  # Q × 40
    "gallery": [
        {
            "index": 0,
            "image": "<dataset>/images/0.jpg",
            "image_path": "/app/input_data/images/<dataset>/images/0.jpg",
        },
        ...
    ],
}
```

Return a dictionary containing **exactly one** of these keys, with a `Q × G` matrix (one row per query, one column per gallery image):

| Key | Values | Ranking direction |
| --- | --- | --- |
| `distances` | Floating-point distances | Smallest first |
| `similarities` | Floating-point scores | Largest first |
| `ranking` | A permutation of gallery indices `0..G-1` in each row | As given, best first |

Lists, NumPy arrays, and tensors are accepted. Invalid shapes, NaN/Inf values, bare arrays, or incomplete ranking permutations fail validation. Equal scores are ordered by gallery position. A practical approach is to predict attribute probabilities for each gallery image once, then compare those predictions with all queries.

## Evaluation

The leaderboard's ranking metric is **mADM** (mean average degree of match). It rewards placing both exact matches and close attribute matches early. Other reported metrics are mAP, Rank-1, Rank-5, Rank-10, and mINP. An exact match requires all 40 attributes to agree with the query.

The latest valid submission, rather than the best previous submission, appears on the leaderboard.

## Dates in the captured challenge description

| Event | Date |
| --- | --- |
| Challenge opens | September 21, 2026 |
| Challenge submission deadline | October 21, 2026, Anywhere on Earth (shown as October 22, 2026 at 18:00 GMT+6 for the current phase) |
| Challenge paper deadline | October 23, 2026, Anywhere on Earth |
| Code and method description deadline | October 30, 2026, Anywhere on Earth |
| RWS workshop at WACV 2027 | January 4–5, 2027 |

After the challenge deadline, teams must provide code, trained models, training code, parameters, and a short method description for reproducibility and winner verification. The organizers also encourage a WACV-format workshop paper.
