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

## Notebooks

For a starter training run on Kaggle, use [upar_track2_resnet18_kaggle.ipynb](notebooks/resnet18_baseline/upar_track2_resnet18_kaggle.ipynb). With Internet and a GPU enabled, it downloads the challenge training data, trains a 40-attribute model, saves `upar_track2_best.pth`, and packages `upar_track2_submission.zip` under `/kaggle/working`.

For a stronger ConvNeXt Tiny run, use [upar-track2-convnext-tiny.ipynb](notebooks/convnext_tiny/upar-track2-convnext-tiny.ipynb). It downloads the same challenge data, fine-tunes an ImageNet-pretrained ConvNeXt Tiny, saves `upar_track2_convnext_tiny_best.pth`, and builds `upar_track2_convnext_tiny_submission.zip`. Training downloads the pretrained weights; the packaged inference code loads only local files and needs no network access.

For an AttriVision-inspired CLIP experiment, use [upar_track2_clip_attrivision_kaggle.ipynb](notebooks/clip_attrivision/upar_track2_clip_attrivision_kaggle.ipynb) or its companion [training script](notebooks/clip_attrivision/upar_clip_attrivision.py). It uses generic pretrained CLIP ViT-B/32 features, positive and negative attribute prompts, and a focal-loss head trained only on the supplied challenge data. It can resume the head from its best checkpoint for up to 100 epochs, selects a checkpoint and ranking score on public validation retrieval mADM, reports full validation metrics, and builds `upar_clip_attrivision_submission.zip`. This is an adaptation of the CLIP and prompt ideas in [AttriVision](https://openaccess.thecvf.com/content/WACV2025W/V3SC/papers/Sedeh_AttriVision_Advancing_Generalization_in_Pedestrian_Attribute_Recognition_using_CLIP_WACVW_2025_paper.pdf), not a reproduction of that paper. An earlier 25-epoch run reached 0.3635 mADM on the public validation gallery. The newer user-reported results appear below; the evaluation split for those numbers was not specified.

For DINOv3 attribute retrieval, use [upar-track2-dinov3-kaggle.ipynb](notebooks/dinov3_attribute/upar-track2-dinov3-kaggle.ipynb) or its companion [training script](notebooks/dinov3_attribute/upar_dinov3.py). It caches frozen DINOv3 ViT-B/16 features, trains a 40-attribute head, selects the checkpoint and ranking method on public validation mADM, and builds `upar_dinov3_submission.zip`. Training requires a Kaggle GPU, Internet access, accepted access to the gated pretrained checkpoint, and a Hugging Face read token supplied through `HF_TOKEN`.

## Reported retrieval results

The ResNet-18 and ConvNeXt Tiny scores below were supplied from their submissions. The CLIP and DINOv3 scores are user-reported results; their evaluation splits were not specified. These are retrieval metrics, not attribute-classification accuracies. Checkpoint epochs and weighted validation BCE values for the first two models come separately from their saved archives.

| Model | mADM (ranking metric) | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 baseline | **0.3469** | 0.1048 | 0.1285 | 0.2232 | 0.2823 | 0.0769 |
| ConvNeXt Tiny | **0.4135** | 0.1413 | 0.1585 | 0.2988 | 0.3767 | 0.1037 |
| CLIP ViT-B/32 + AttriVision-inspired attribute head | **0.4278** | 0.1450 | 0.1471 | 0.3253 | 0.4253 | 0.1060 |
| DINOv3 ViT-B/16 + pooled attribute head | **0.4832** | 0.2298 | 0.2396 | 0.4393 | 0.5438 | 0.1686 |

ConvNeXt Tiny improves mADM by **0.0666** over the reported ResNet-18 result.

## DINOv3 ViT-B/16: configuration and method

The [Kaggle notebook](notebooks/dinov3_attribute/upar-track2-dinov3-kaggle.ipynb) uses the companion [training script](notebooks/dinov3_attribute/upar_dinov3.py) to train only an attribute head on frozen, generic pretrained DINOv3 features. The [saved submission archive](notebooks/dinov3_attribute/upar_dinov3_submission.zip) contains `run.py`, the example submission's unchanged `metadata.yaml`, a traced encoder and head in `assets/model.pt`, and `assets/config.json`. The archive records the 40 attribute names and `"score_method": "nll"`.

| Component | Setting |
| --- | --- |
| Training and validation | Official `train.csv` (97,669 images) and `val.csv` (33,407 images), with 40 binary labels in the fixed challenge order; head training uses only the supplied challenge data |
| Backbone | `facebook/dinov3-vitb16-pretrain-lvd1689m`, loaded with Hugging Face `AutoModel`; backbone stays frozen in evaluation mode |
| Input | RGB, resized to **256 × 128** pixels (height × width); ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`; no training augmentation |
| Image features | Concatenate the CLS token, mean of all patch tokens, mean of the upper-half patch grid, and mean of the lower-half patch grid; exclude register tokens from patch pooling |
| Trainable head | LayerNorm over the concatenated features, followed by a linear layer producing 40 independent attribute logits |
| Loss and optimizer | Unweighted `binary_cross_entropy_with_logits`; AdamW, learning rate `1e-3`, weight decay `0.01`; no learning-rate schedule |
| Training | Cache frozen image features in half precision; feature batch size 64, head batch size 2,048, at most 60 epochs, seed 42, 2 feature-loader workers; CUDA autocast during feature extraction |
| Checkpoint selection | NLL retrieval mADM on a fixed random subset of at most 256 unique validation queries, checked at epoch 1, every 5 epochs, and epoch 60; save the best head as `upar_dinov3_head.pth`; stop after 3 consecutive checks without improvement |
| Recorded run | Stopped at epoch 55 and restored epoch **40**, with selection mADM **0.5927** |
| Ranking selection | Compare NLL and probability L1 distances on the same selected queries; choose the higher-mADM method, then evaluate all unique validation queries; this run selected NLL |
| Inference | Offline TorchScript encoder and head, batches of 64 images, attribute columns reordered to the incoming query names; return a `Q × G` `distances` matrix, with lower distances ranking first |

For gallery image $g$, let $c_g$ be its CLS token and $P_g$ its final patch-token grid. With $P_g^{\mathrm{upper}}$ and $P_g^{\mathrm{lower}}$ denoting the top and bottom halves of that grid, the feature and predicted logits are

$$
f_g=\operatorname{concat}\left(c_g,\operatorname{mean}(P_g),\operatorname{mean}(P_g^{\mathrm{upper}}),\operatorname{mean}(P_g^{\mathrm{lower}})\right),\qquad
z_g=W\operatorname{LayerNorm}(f_g)+b.
$$

The regional averages give the head separate upper-body and lower-body information alongside the global features. Only LayerNorm and the linear head receive optimizer updates. For training labels $y_{ia}\in\{0,1\}$ and probabilities $p_{ia}=\sigma(z_{ia})$, the loss for a batch of $B$ images is

$$
L_B=-\frac{1}{40B}\sum_{i=1}^{B}\sum_{a=1}^{40}\left[y_{ia}\log p_{ia}+(1-y_{ia})\log(1-p_{ia})\right].
$$

For binary query $q$, the packaged submission uses the independent-label Bernoulli negative log likelihood:

$$
d_{\mathrm{NLL}}(q,g)=\sum_{a=1}^{40}\left[\log(1+e^{z_{ga}})-q_a z_{ga}\right].
$$

The alternative considered during validation is $d_{\mathrm{L1}}(q,g)=\sum_{a=1}^{40}|q_a-\sigma(z_{ga})|$. Both distances are sorted in ascending order. Export fixes the rotary positional embedding buffers for the configured input size and checks traced predictions and packaged inference on CPU and GPU, allowing submission inference without Transformers or network access.

The saved notebook separately reports these **public-validation** metrics for the restored head and NLL ranking:

| mADM | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5898 | 0.1797 | 0.1984 | 0.3850 | 0.4685 | 0.0934 |

The **0.4832 mADM** result in the main results table is the user-reported evaluation result, whose split was not specified; it is separate from the public-validation output above.

## CLIP ViT-B/32 with an AttriVision-inspired attribute head

The [Kaggle notebook](notebooks/clip_attrivision/upar_track2_clip_attrivision_kaggle.ipynb) downloads the official challenge data and runs the companion [training script](notebooks/clip_attrivision/upar_clip_attrivision.py). The [submission archive](notebooks/clip_attrivision/upar_clip_attrivision_submission.zip) contains a traced image encoder and head, `run.py`, `metadata.yaml`, and `assets/config.json`. The archive's config records the 40 attribute names and `"score_method": "nll"`. This is an adaptation inspired by AttriVision's use of CLIP and attribute prompts, not a reproduction of its architecture or training protocol.

| Component | Setting |
| --- | --- |
| Training and validation | Official `train.csv` and `val.csv` with 40 binary labels in the fixed challenge order; no external pedestrian-attribute training data |
| Backbone | `openai/clip-vit-base-patch32`; pretrained CLIP vision and text encoders are frozen; image embeddings come from the vision pooled output followed by CLIP's visual projection and L2 normalization |
| Input | RGB, padded to 224 × 224 with black pixels while preserving aspect ratio; CLIP mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)` |
| Text features | One positive and one negative prompt per attribute; each CLIP text embedding is L2 normalized, then the negative embedding is subtracted from the positive embedding |
| Trainable head | A 512-to-40 linear residual plus one trainable log scale; residual weights and biases start at zero, log scale starts at 2.3 |
| Loss and optimizer | Focal binary cross-entropy with gamma 2; AdamW, learning rate `1e-3`, weight decay `0.01`; no learning-rate schedule |
| Training | Frozen image features cached as half precision; head batch size 2,048, up to 100 total epochs, seed 42, 2 data-loader workers; image feature batch size 128; CUDA autocast during feature extraction |
| Model selection | Check validation retrieval mADM at epoch 1, every 5 epochs, and epoch 100 on a fixed random subset of at most 256 unique validation queries; save the best head to `upar_clip_attrivision_head.pth` and resume from it if present |
| Ranking selection | Compare negative log likelihood (NLL) and L1 distances on the same selected queries; choose the method with higher mADM, then report all unique validation queries; the packaged artifact uses NLL |
| Inference | TorchScript model predicts 40 logits in batches of 128, reorders columns by the requested attribute names, and returns a `Q × G` `distances` matrix; lower distances rank first |

Let $x_g$ be gallery image $g$. Let $v_g$ be its projected and L2-normalized CLIP image feature, and let $t_a^+$ and $t_a^-$ be the L2-normalized CLIP text features for positive and negative prompts for attribute $a$. The frozen prompt difference and trainable logit are

$$
\Delta_a=t_a^+-t_a^-,\qquad
z_{ga}=e^s v_g^\top\Delta_a+w_a^\top v_g+b_a,
$$

where $s$, $w_a$, and $b_a$ are trained, with $s=2.3$, $w_a=0$, and $b_a=0$ at initialization. The prompt differences are computed once and kept fixed. Only the residual layer and log scale receive optimizer updates.

For label $y_{ia}\in\{0,1\}$ and predicted probability $p_{ia}=\sigma(z_{ia})$, the focal loss for a batch of $B$ images is

$$
\ell_{ia}=-y_{ia}\log p_{ia}-(1-y_{ia})\log(1-p_{ia}),\qquad
L_B=\frac{1}{40B}\sum_{i=1}^{B}\sum_{a=1}^{40}(1-e^{-\ell_{ia}})^2\ell_{ia}.
$$

For query $q\in\{0,1\}^{40}$, the packaged submission ranks gallery image $g$ using the independent-label Bernoulli NLL:

$$
d_{\mathrm{NLL}}(q,g)
=\sum_{a=1}^{40}\bigl[\log(1+e^{z_{ga}})-q_a z_{ga}\bigr]
=\sum_{a=1}^{40}\log(1+e^{z_{ga}})-q^\top z_g.
$$

The script also considers probability L1 distance during validation:

$$
d_{\mathrm{L1}}(q,g)=\sum_{a=1}^{40}|q_a-p_{ga}|
=\sum_{a=1}^{40}p_{ga}+\sum_{a=1}^{40}q_a(1-2p_{ga}).
$$

Both distances are sorted in ascending order. Attribute columns are aligned to the incoming query names before scoring. The full definitions of mADM, mAP, Rank-K, and mINP are in [Evaluation](#evaluation). The six scores in the results table were supplied by the user; the archive does not record their evaluation split or the selected checkpoint epoch.

## ResNet-18 baseline: configuration

The [saved ResNet-18 submission archive](notebooks/resnet18_baseline/upar_track2_submission.zip) contains a checkpoint recording epoch **2** and weighted validation BCE **0.4285**.

The notebook and its companion [training script](train_upar_track2_kaggle.py) use this configuration:

| Component | Setting |
| --- | --- |
| Training data | Supplied `train.csv`: 97,669 images from PA-100K, Market1501, and PETA |
| Validation data | Supplied `val.csv`: 33,407 images; used to select the checkpoint by attribute loss |
| Backbone | `torchvision.models.resnet18` with default ImageNet pretrained weights; all layers fine-tuned |
| Output | Final 512-to-40 fully connected layer gives 40 independent attribute logits |
| Input | RGB, resized to **256 × 128** pixels (height × width); normalized with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)` |
| Training augmentation | Random horizontal flip with the torchvision default probability of 0.5; no other augmentation in the notebook |
| Loss | `BCEWithLogitsLoss` with a per-attribute positive-class weight, defined below |
| Optimizer | AdamW; learning rate `3e-4`, weight decay `1e-4`; no learning-rate schedule |
| Run | 5 epochs, batch size 128, 2 data-loader workers, seed 42; CUDA mixed precision when available |
| Checkpoint | Epoch with the lowest mean validation weighted BCE; saved as `upar_track2_best.pth` |
| Inference | Batch size 128, sigmoid probabilities, unweighted L1 query distance; returns `distances` |

For attribute $a$, the positive-class weight is

$$
w_a=\min\left(20,\max\left(1,\frac{N-n_a}{\max(n_a,1)}\right)\right).
$$

Here $N$ is the number of training images and $n_a$ is the number labeled positive for attribute $a$. Training minimizes the average weighted binary cross-entropy over the 40 logits. The notebook **does not select or calibrate the checkpoint using retrieval mADM**.

At inference, $q_a\in\{0,1\}$ is the query value, and $p_{ga}=\sigma(z_{ga})$ is the predicted probability for attribute $a$ of gallery image $g$. The returned distance is

$$
d(q,g)=\sum_{a=1}^{40}|q_a-p_{ga}|.
$$

Equivalently,

$$
d(q,g)=\sum_{a=1}^{40}p_{ga}+\sum_{a=1}^{40}q_a(1-2p_{ga}).
$$

Gallery images are ranked by **ascending** distance. The second form is the vectorized computation in `run.py`. Attribute names are reordered to match the query before scoring. The positive-class weighting helps rare labels during training but can shift probability calibration; the baseline does not correct for that shift before applying the L1 distance.

## ConvNeXt Tiny: configuration and scoring

The [ConvNeXt Tiny notebook](notebooks/convnext_tiny/upar-track2-convnext-tiny.ipynb) uses the same training and validation images and attribute order as the baseline. Its [saved submission archive](notebooks/convnext_tiny/upar_track2_convnext_tiny_submission.zip) records checkpoint epoch **2** and weighted validation BCE **0.3756** (full value `0.37556340439679947`). Checkpoint selection uses weighted validation BCE, not retrieval mADM.

| Component | Setting |
| --- | --- |
| Training data | Supplied `train.csv`: 97,669 images from PA-100K, Market1501, and PETA |
| Validation data | Supplied `val.csv`: 33,407 images; used to select the checkpoint by attribute loss |
| Backbone | `torchvision.models.convnext_tiny` with `ConvNeXt_Tiny_Weights.DEFAULT` ImageNet weights during training; all layers fine-tuned |
| Output | Replace `model.classifier[-1]` with a linear layer producing 40 independent attribute logits |
| Input | RGB, resized to **256 × 128** pixels (height × width); normalized with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)` |
| Training augmentation | Random horizontal flip with probability 0.5; no other augmentation |
| Loss | `BCEWithLogitsLoss` with the per-attribute positive-class weights below |
| Optimizer and schedule | AdamW, initial learning rate `1e-4`, weight decay `1e-4`; `CosineAnnealingLR` stepped after each epoch with `T_max=8` |
| Run | 8 epochs, batch size 64, 2 data-loader workers, seed 42; CUDA mixed precision when available |
| Checkpoint | Epoch with the lowest mean validation weighted BCE; saved as `upar_track2_convnext_tiny_best.pth` |
| Inference | Batch size 64, sigmoid probabilities, unweighted L1 query distance; returns `distances` |

For training image $i$ and attribute $a$, let $y_{ia}\in\{0,1\}$ be the label, $z_{ia}$ the model logit, $p_{ia}=\sigma(z_{ia})$ the predicted probability, $N$ the training-image count, and $n_a$ the count of positive labels for attribute $a$. The positive weight and mean training loss for a batch of $B$ images are

$$
w_a=\min\left(20,\max\left(1,\frac{N-n_a}{\max(n_a,1)}\right)\right),
$$

$$
L_B=-\frac{1}{40B}\sum_{i=1}^{B}\sum_{a=1}^{40}\left[w_a y_{ia}\log(p_{ia})+(1-y_{ia})\log(1-p_{ia})\right].
$$

At inference, for binary query $q$ and gallery image $g$, the notebook predicts $p_{ga}=\sigma(z_{ga})$ once per image and computes

$$
d(q,g)=\sum_{a=1}^{40}|q_a-p_{ga}|=\sum_{a=1}^{40}p_{ga}+\sum_{a=1}^{40}q_a(1-2p_{ga}).
$$

Gallery images are ranked by **ascending** distance. Attribute columns are reordered to match the query names. The inference code does not calibrate probabilities or adjust the 40 attribute weights for ranking. The ZIP contains `run.py`, the unchanged `metadata.yaml`, and `assets/upar_track2_convnext_tiny_best.pth`; `run.py` constructs the model with `weights=None` and loads that local checkpoint.

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

For one query, let $G$ be the gallery size, $R$ the number of exact matches, and $r_1<\cdots<r_R$ their one-based positions in the returned ranking. The per-query metrics are

$$
AP=\frac{1}{R}\sum_{k=1}^{R}\frac{k}{r_k},
$$

$$
INP=\frac{R}{r_R},
$$

$$
Rank_K=[r_1\leq K],\qquad K\in\{1,5,10\},
$$

where the bracket is $1$ when its condition is true and $0$ otherwise. To compute ADM, let $g_i$ be the image at rank $i$ and $y_{ga}$ the true value of attribute $a$ for gallery image $g$. Define $D_q(g)$ as its degree of match, $\mu_q$ as the gallery mean, $M_q$ as the maximum degree, and $\widetilde D_q(g)$ as the normalized degree:

$$
D_q(g)=\frac{1}{40}\sum_{a=1}^{40}[y_{ga}=q_a],
$$

$$
\mu_q=\frac{1}{G}\sum_{g=1}^{G}D_q(g),
$$

$$
M_q=\max_{1\leq g\leq G}D_q(g),
$$

$$
\widetilde D_q(g)=\max\left(0,\frac{D_q(g)-\mu_q}{M_q-\mu_q}\right).
$$

Each challenge query has at least one exact gallery match, so $M_q=1$. The denominator is positive when the gallery also contains a nonmatch. At every exact-match rank, ADM credits the average normalized agreement of *all* images retrieved up to that rank:

$$
ADM_q=\frac{1}{R}\sum_{k=1}^{R}\left[\frac{1}{r_k}\sum_{i=1}^{r_k}\widetilde D_q(g_i)\right].
$$

**mADM, mAP, and mINP** are the corresponding per-query values averaged over all queries. **Rank-K** is the average of its per-query indicator. Thus mADM also rewards near matches early in the list, while mAP and Rank-K count only exact 40-attribute matches. These are evaluator equations; the ResNet-18 notebook does not compute them during training.

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
