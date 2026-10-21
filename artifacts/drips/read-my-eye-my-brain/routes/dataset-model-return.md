# Route return: a real dataset and a real pretrained model, run end to end

One line: the APTOS 2019 fundus images are real and downloaded (4 images, viewed directly), and a real pretrained diabetic-retinopathy-grading ResNet-18 ran inference on them end to end with its published weights, no training and no fabrication anywhere in this chain. Kaggle itself was not reachable (no API key configured on this machine); the dataset came from a public Hugging Face mirror instead.

Everything below ran in `/home/molef/playground/deep-learning/my-eye-my-brain/dataset-smoke-test/`.

## 1. Dataset: APTOS 2019, via a Hugging Face mirror (Kaggle blocked)

**Blocked: Kaggle needs an API key that is not configured here.** `pip3 show kaggle` found no `kaggle` package installed, and there is no `~/.kaggle/kaggle.json` and no `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables. (There is an unrelated file at `~/.kaggle/access_token`, but it is not a Kaggle credentials file and was not used.) Rather than stall there, I searched Hugging Face's dataset hub instead, per the task's own fallback instruction.

**Ran:** found `EslamHasan/APTOS2019DiabeticRetinopathy` on the Hugging Face Hub, a 1.9K-row parquet mirror of the real Kaggle APTOS 2019 Blindness Detection competition images (5.0 GB total across 9 parquet shards, too large to pull whole for a smoke test). Instead of downloading the full parquet, I pulled 4 individual full-resolution source images directly through the Hugging Face datasets-viewer's cached-asset URLs (these are the same JPEGs backing the parquet rows, served as plain files):

```
curl -sL -o images/aptos_0.jpg "https://datasets-server.huggingface.co/cached-assets/EslamHasan/APTOS2019DiabeticRetinopathy/.../default/train/0/image/image.jpg?...signed..."
curl -sL -o images/aptos_1.jpg "...row 1..."
curl -sL -o images/aptos_2.jpg "...row 2..."
curl -sL -o images/aptos_3.jpg "...row 3..."
```

(The exact signed URLs are in the Hugging Face dataset viewer API response for `EslamHasan/APTOS2019DiabeticRetinopathy`, config `default`, split `train`, rows 0-4; they expire, so a reader reproducing this should re-query `https://datasets-server.huggingface.co/rows?dataset=EslamHasan%2FAPTOS2019DiabeticRetinopathy&config=default&split=train&offset=0&length=5` for fresh signed links rather than reusing the ones above.)

This landed 4 real JPEGs in `images/`, 205-313 KB each, at their original camera resolution (2416x1736 and 3216x2136), confirmed with `file images/*.jpg`.

**What they look like.** I opened two of the four directly:

- `images/aptos_0.jpg` (2416x1736): a circular fundus photograph on a black surround. The optic disc sits left of center as a bright orange-white circular patch with the retinal blood vessels branching out from it. A faint grey-white hazy patch sits right of center, roughly where the macula would be.
- `images/aptos_2.jpg` (3216x2136): a higher-resolution fundus photograph of the same kind, optic disc at left (brighter, whiter here), vessels radiating outward and curving across the retina, with a small darker reddish patch near the center. A sharper, clearer image than `aptos_0.jpg`.

Both are genuine retinal camera photographs, not synthetic or placeholder images: the circular field-of-view vignette, radiating vessel pattern, and optic-disc glare are all real camera artifacts.

**What was not resolved:** the HF ClassLabel schema on this mirror lists `["Mild","Moderate","No DR","Proliferative DR","Severe"]` in alphabetical order and all 4 rows I fetched carry a raw label index of 0. The original APTOS 2019 competition's label scale is a different, non-alphabetical order (0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR), so whether this mirror's `0` means "No DR" (the original ordering) or "Mild" (its own alphabetical `ClassLabel` ordering) is ambiguous from the files available, and I did not resolve it. Treat any ground-truth DR grade for these 4 images as unknown, not as "Mild" or "No DR".

## 2. Model: a real pretrained diabetic-retinopathy-grading ResNet-18

**Ran:** `ClementP/FundusDRGrading-resnet18` on the Hugging Face Hub, a `timm` ResNet-18 (11.2M parameters) released under the MIT license, explicitly trained on APTOS + EYEPACS + IDRiD + DDR for DR grading, not gated, weights published as `model.safetensors` (part of a `ClementP/FundusDRGrading-*` family that also has EfficientNet, ConvNeXt, Swin, DINOv2-ViT and other backbones on the same task if a reader wants a heavier model).

Set up a Python virtual environment in the playground folder (the system Python is externally managed and blocks `pip install`), then installed CPU-only PyTorch, `timm`, `safetensors`, `huggingface_hub`, `torchvision`, and `pillow`:

```
cd /home/molef/playground/deep-learning/my-eye-my-brain/dataset-smoke-test
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install timm safetensors huggingface_hub pillow torchvision \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Then ran `run_inference.py` (written into the same folder; loads the published `config.json` and `model.safetensors` straight from the Hub, builds the matching `timm` ResNet-18 skeleton, and runs the 4 real images through it):

```
.venv/bin/python run_inference.py
```

**This actually executed and printed real numbers, state-dict load confirmed clean (`Missing keys: []`, `Unexpected keys: []`, meaning the downloaded weights match the architecture exactly, not a partial or mismatched load):**

```
Running inference on 4 real APTOS fundus images:
  images/aptos_0.jpg: raw model output = 0.5715  (size (2416, 1736))
  images/aptos_1.jpg: raw model output = 0.2231  (size (2416, 1736))
  images/aptos_2.jpg: raw model output = 0.1974  (size (3216, 2136))
  images/aptos_3.jpg: raw model output = 0.8440  (size (2416, 1736))
```

**What this output means, and what it does not.** The published `config.json` sets `num_classes: 1`, i.e. this is a single-scalar regression head, the standard way APTOS-competition models predict the ordinal 0-4 severity scale (round the scalar to the nearest integer in [0, 4] for a discrete grade). The model card does not publish the exact train-time scaling or rounding convention, so I report the raw scalar rather than inventing a grade label or a confidence percentage for it. Read qualitatively: `aptos_3.jpg` scored highest (0.84) and `aptos_2.jpg` lowest (0.20) among the 4, which is a real, reproducible difference in model output, not a claim about which image has more disease (that would need the correct rounding/calibration convention, which is not published).

## 3. Reproduce this

```
cd /home/molef/playground/deep-learning/my-eye-my-brain/dataset-smoke-test
python3 -m venv .venv
.venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --quiet timm safetensors huggingface_hub pillow torchvision \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# images/aptos_0.jpg .. aptos_3.jpg are already in the folder;
# to re-fetch fresh ones, query the HF datasets-viewer rows endpoint for fresh signed URLs:
#   curl -s "https://datasets-server.huggingface.co/rows?dataset=EslamHasan%2FAPTOS2019DiabeticRetinopathy&config=default&split=train&offset=0&length=5"
.venv/bin/python run_inference.py
```

Total download footprint for this whole smoke test: 4 JPEGs at 205-313 KB each (under 1 MB total) plus the model's `config.json` and `model.safetensors` (well under 100 MB; the 5 GB APTOS parquet mirror was never downloaded in full). The `.venv` itself is about 1.1 GB on disk (CPU PyTorch plus dependencies) and runs on CPU only, no GPU needed.

## What could not be done and why

**Kaggle itself:** blocked, no API key configured on this machine (see section 1). The Hugging Face mirror substituted for it; the images themselves are the same APTOS 2019 photographs, so nothing about "what a fundus photo looks like" was lost by taking this route.

**The exact train-time label convention for the ResNet-18's scalar output:** not published on the model card, so I report the raw number rather than a fabricated class name or confidence score.

## Changes for the walk: clinic_day.py

`clinic_day.py`'s two-by-two table, built on invented numbers, now has one real leg under it: a real fundus photograph (`images/aptos_0.jpg` through `aptos_3.jpg`, genuine APTOS 2019 camera images) and a real pretrained model's real output on that photograph (`ClementP/FundusDRGrading-resnet18`, raw scalar scores 0.20 to 0.84 across the 4 images, reproducible with the commands above). The sensitivity/specificity numbers in the two-by-two table itself are still invented: turning this into a real table would need the correct ground-truth grade for each image (unresolved here, see section 1) and enough images to compute a confusion matrix, neither of which this smoke test attempted.
