# DictionaryBank: reproducible code

This repository contains the code needed to reproduce the main
DictionaryBank experiments:

- frozen CLIP, SigLIP and DINOv2 feature extraction;
- independent content-only Top-k sparse dictionaries;
- one global Top-64 support over three independently trained dictionaries;
- source-domain-centered class prototypes;
- DomainNet leave-one-domain-out training and evaluation;
- the RVL-CDIP secondary transfer experiment;
- the ensemble, raw-feature, support-allocation and latency controls.

No datasets, pretrained weights, sparse-autoencoder checkpoints or paper files
are included in this repository.

## Method

For each frozen encoder stream \(s\), three content-only SAEs are trained
independently. Each member has 4,096 latent candidates and uses a Top-64
support during training. The encoder matrices are concatenated into a
12,288-candidate bank. At inference, one global Top-64 operation selects a
shared support over the union of the three member dictionaries. ReLU is applied
to the selected logits.

Source-domain means are removed from class means before the centered class
prototypes are computed. Target batches do not update the dictionary, center or
prototype. The three encoder streams produce separate class probabilities,
which are averaged with equal weights.

The deployment path therefore uses one sparse execution and at most 64
nonzero selected entries per encoder stream, rather than three separate
member executions and 192 selected entries.

## Repository layout

```text
autoencoder/                         model, training and diagnostic code
experiments/09_cviu_controls/        dictionary-bank evaluator and preparation
experiments/10_rvl_q1_controls/      RVL-CDIP analysis
experiments/11_reviewer_controls/    ensemble, raw and efficiency controls
experiments/results/                  selected JSON summaries reported in the paper
data_prepare/                        public-weight feature extraction
scripts/                             portable experiment launchers
make_figures.py                      regenerate the data figures from result JSON
```

## Environment

Python 3.10 or newer and an NVIDIA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

The released results used PyTorch 2.7.1, CUDA 11.8 and an NVIDIA RTX A6000.
A6000-class memory is recommended for the complete DomainNet protocol.

## Public model weights

Download the public weights from Hugging Face or the official model sources:

```bash
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir weights/clip-vit-large-patch14
huggingface-cli download google/siglip-base-patch16-224 \
  --local-dir weights/siglip-base-patch16-224
huggingface-cli download facebook/dinov2-base \
  --local-dir weights/dinov2-base
```

The scripts accept either Hugging Face model identifiers or local snapshot
directories. No third-party model weights are redistributed here.

## Public datasets

Download DomainNet from its official release
(`http://ai.bu.edu/M3SDA/`) and RVL-CDIP from the official CMU dataset page
(`https://www.cs.cmu.edu/~aharley/rvl-cdip/`). Keep the datasets outside this
repository. RVL-CDIP archives and DomainNet image files must not be committed
to Git.

DomainNet is expected as:

```text
DomainNet/
  train/<domain>/<class>/<image files>
  test/<domain>/<class>/<image files>
```

The six domain names are `clipart`, `infograph`, `painting`, `quickdraw`,
`real` and `sketch`.

## 1. Extract frozen features

Generate the training features first:

```bash
python data_prepare/prepare_domainnet_features.py \
  --split-dir /path/to/DomainNet/train \
  --out cache/domainnet/train.pt \
  --metadata cache/domainnet/train_metadata.json
```

Generate the test features with the training class mapping:

```bash
python data_prepare/prepare_domainnet_features.py \
  --split-dir /path/to/DomainNet/test \
  --out cache/domainnet/test.pt \
  --metadata cache/domainnet/test_metadata.json \
  --label-map cache/domainnet/train_metadata.json
```

Features are L2-normalized and stored with integer labels, domain identifiers
and source paths. To verify the download pipeline without a full run, add
`--limit-per-class 5`.

The repository does not redistribute these caches. Regenerate them from the
public image datasets and public pretrained weights.

## 2. Build DomainNet leave-one-domain-out folds

For target domain identifier \(t\in\{0,\ldots,5\}\):

```bash
python autoencoder/domainnet_fold_cache.py \
  --train-cache cache/domainnet/train.pt \
  --test-cache cache/domainnet/test.pt \
  --target-domain "$t" \
  --out-dir cache/domainnet/folds
```

The mapping is:

```text
0 clipart  1 infograph  2 painting  3 quickdraw  4 real  5 sketch
```

Each fold contains only the label-compatible classes shared by the source
domains and the target domain.

## 3. Train the three independent content SAEs

For every target and seed `0,1,2`:

```bash
python autoencoder/pacs_factor_sae.py \
  --cache cache/domainnet/folds/target${t}_train.pt \
  --target-cache cache/domainnet/folds/target${t}_test.pt \
  --target-domain "$t" \
  --streams clip siglip dino \
  --content-latents 4096 \
  --content-k 64 \
  --nuisance-k 0 \
  --epochs 20 \
  --batch-size 512 \
  --class-weight 2 \
  --classifier-kind cosine \
  --domain-weight 0 \
  --adversarial-weight 0 \
  --alignment-weight 0.1 \
  --contrastive-weight 0.1 \
  --seed "$seed" \
  --out outputs/domainnet/target${t}/seed${seed}.pt \
  --report-json outputs/domainnet/target${t}/seed${seed}.json
```

## 4. Evaluate individual members and DictionaryBank

Evaluate each of the three single members with Top-64:

```bash
python experiments/09_cviu_controls/evaluate_bank.py \
  --cache cache/domainnet/folds/target${t}_train.pt \
  --target-cache cache/domainnet/folds/target${t}_test.pt \
  --checkpoints outputs/domainnet/target${t}/seed${seed}.pt \
  --target-domain "$t" \
  --active-budget 64 \
  --out outputs/domainnet/eval/target${t}_single_seed${seed}.pt \
  --report-json outputs/domainnet/eval/target${t}_single_seed${seed}.json
```

Evaluate the full bank over all three checkpoints:

```bash
python experiments/09_cviu_controls/evaluate_bank.py \
  --cache cache/domainnet/folds/target${t}_train.pt \
  --target-cache cache/domainnet/folds/target${t}_test.pt \
  --checkpoints \
    outputs/domainnet/target${t}/seed0.pt \
    outputs/domainnet/target${t}/seed1.pt \
    outputs/domainnet/target${t}/seed2.pt \
  --target-domain "$t" \
  --active-budget 64 \
  --out outputs/domainnet/eval/target${t}_bank_full.pt \
  --report-json outputs/domainnet/eval/target${t}_bank_full.json
```

Pairs are evaluated by passing the corresponding two checkpoints. The ordinary
ensemble is obtained by averaging the three single-member probability tensors.

After the feature caches and fold caches exist, the complete DomainNet control
pipeline can also be launched with:

```bash
bash scripts/run_domainnet_controls.sh
```

Set `CACHE_DIR` and `OUTPUT_DIR` to relocate caches and results.

## 5. Recompute paper statistics

After all six targets have the same file names used by
`analyze_ensembles.py`, run:

```bash
python experiments/11_reviewer_controls/analyze_ensembles.py \
  --root outputs/domainnet \
  --bootstrap-samples 10000
```

This script checks label alignment, computes the macro and pooled paired
comparisons, and writes `ensemble_analysis.json`.

## RVL-CDIP secondary transfer

The RVL-CDIP experiment starts by extracting features from the public dataset.
Use `data_prepare/rvl_large_cache.py` to create `train.pt` and `test.pt`, then
run `experiments/09_cviu_controls/prepare_rvl.py` to combine them with explicit
`source` and `target` domain tags.

The feature extractor used for the reported subset is based on the public
Hugging Face dataset `SBakkali/rvlcdip_subset`:

```bash
huggingface-cli download SBakkali/rvlcdip_subset \
  --repo-type dataset \
  --local-dir data/rvlcdip_subset
```

```bash
python data_prepare/rvl_large_cache.py \
  --data-dir data/rvlcdip_subset/data \
  --out-dir cache/rvl_cdip \
  --siglip-model weights/siglip-base-patch16-224 \
  --clip-model weights/clip-vit-large-patch14 \
  --dino-model weights/dinov2-base

python experiments/09_cviu_controls/prepare_rvl.py \
  --train cache/rvl_cdip/train.pt \
  --test cache/rvl_cdip/test.pt \
  --out cache/rvl_cdip/rvl400.pt
```

Before training, verify the source and target counts and the content overlap:

```bash
python data_prepare/rvl_hash_audit.py \
  --corpus source cache/rvl_cdip/source_parquet \
  --corpus target cache/rvl_cdip/target_parquet \
  --out outputs/rvl_hash_audit.json
```

The secondary split used in the paper contains 6,385 source and 1,600 target
images over 16 classes. If the public dataset version yields different counts,
preserve the actual manifest and report the difference; do not relabel or
silently substitute a different split.

Once `CACHE` points to the verified combined cache, run:

```bash
bash scripts/run_rvl_q1.sh
```

## Expected headline results

On the complete DomainNet leave-one-domain-out protocol:

| Method | Macro accuracy |
|---|---:|
| Three-model probability ensemble | 67.991% |
| DictionaryBank, Top-64 | 67.971% |

The bank matches the ensemble within 0.020 percentage points and uses one
sparse execution with at most 64 nonzero selected entries per stream instead
of 192 entries across three executions.

The selected JSON summaries used by the paper are included under
`experiments/results/`. They contain metrics, not image data or checkpoints.
After a completed rerun, figures can be regenerated with:

```bash
python make_figures.py
```

## Reproducibility notes

- Seeds control member initialization, source fitting/validation splits and
  mini-batch order.
- Target labels are used after prediction for evaluation and for defining the
  label-compatible class inventory. They do not update model parameters,
  source centers, prototypes or checkpoint selection.
- The Wide SAE used as an additional control is trained at Top-192 and
  evaluated at Top-64. It is not a training-matched Top-64 SAE.
- No source-target content overlap is allowed.
- All headline numbers should be checked against the generated JSON artifacts.

## License

Code is released under the MIT License. Dataset images and pretrained model
weights retain their original licenses and are not redistributed.
