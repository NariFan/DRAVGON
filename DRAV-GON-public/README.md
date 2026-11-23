# DRAV-GON

Research code for **DRAV-GON: Structural Symmetry in Cross-Modal Interaction for Robust Audio-Visual Emotion Recognition**.

The model combines MFCC, CQT, and frozen BEATs or emotion2vec+ embeddings with an EfficientFace visual encoder, bidirectional cross-modal interaction, feature-wise Gumbel-noise sigmoid gating, and three jointly optimized classification heads (`M`, `A`, and `V`).

## What is included

- Training, validation, and testing entry points.
- RAVDESS and PKU-ER preprocessing scripts.
- Waveform-noise and modality-corruption evaluation.
- The 16-configuration, five-fold RAVDESS ablation runner (80 runs).
- Five best-validation checkpoints for the independently rerun full-reference P1-1 model, under `checkpoints/full_model/`.
- The original MIT license in `LICENSE`.

**Not included:** datasets, third-party pretrained weights, feature caches, generated annotations, logs, other model checkpoints, experimental result directories, manuscript files, or reviewer-response files. These remain external to this source-code release.

## Environment

The verified configuration is Python 3.10, PyTorch 2.2.2, and CUDA 11.8. A CPU is sufficient for the data-free checks, but full video training is intended for a CUDA GPU.

```bash
cd DRAV-GON-public
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a PyTorch build compatible with your local CUDA driver. `funasr` and `modelscope` are needed only for emotion2vec+. MTCNN preprocessing additionally requires `facenet-pytorch`, and `prepare_raw_layout.py` requires `imageio-ffmpeg`.

## Pretrained weights

Create these directories locally and download the weights from their original providers:

| Local path | Source | Required |
|---|---|---|
| `pre-trained_checkpoints/BEATs_iter3.pt` | [Microsoft BEATs](https://github.com/microsoft/unilm/tree/master/beats) | For `--audio_backbone beats` |
| `pre-trained_checkpoints/emotion2vec/` | [emotion2vec+ on Hugging Face](https://huggingface.co/emotion2vec/emotion2vec_plus_base) | For `--audio_backbone emotion2vec`; include `model.pt`, `config.yaml`, `configuration.json`, and `tokens.txt` |
| `Pretrain-model/EfficientFace_Trained_on_AffectNet7.pth.tar` | [EfficientFace](https://github.com/zengqunzhao/EfficientFace) | Optional; use `--pretrain_path None` to disable initialization from this file |

Third-party pretrained weights and partial downloads are excluded by `.gitignore`. Run all commands from the repository root because the default checkpoint paths are relative to the current working directory.

## Included full-reference checkpoints

The directory `checkpoints/full_model/ravdess-30f-e2v_gs_it-h1-nit_har-full_dir-both-d1_P1-1/` contains `best.pth` under `fold0/` through `fold4/`. These are the best-validation checkpoints from the independently rerun P1-1 full-reference configuration in the 80-run ablation batch. They are not the earlier main-table checkpoints. The five-fold result for this independent P1-1 batch is 91.32% ± 3.53% mean accuracy.

Each checkpoint is approximately 16.96 MiB, for a total of approximately 84.8 MiB. The five files are explicitly included by `.gitignore` and are each below GitHub's 100 MB single-file limit.

To evaluate these checkpoints on clean held-out samples after preparing the data and the emotion2vec+ weights:

```bash
python main.py --no_train --no_val \
  --dataset RAVDESS \
  --annotation_path ravdess_preprocessing/annotations_fold30f.txt \
  --n_folds 5 --n_classes 8 --sample_duration 30 \
  --audio_backbone emotion2vec \
  --fusion gs_it --num_heads 1 --har full --cross_dir both \
  --mask noise --batch_size 4 --tag P1-1 \
  --pretrain_path None --result_path checkpoints/full_model \
  --n_threads 8 --test_only_both
```

`--pretrain_path None` skips external EfficientFace initialization before the selected full checkpoint is loaded. The evaluation uses the same fold-specific `best.pth` files.

## Data preparation

RAVDESS is available at [doi:10.5281/zenodo.1188976](https://doi.org/10.5281/zenodo.1188976). PKU-ER is not redistributed here; please follow the access procedure of its original dataset authors.

The preprocessing scripts use environment variables instead of machine-specific paths:

```bash
export RAVDESS_ROOT=/absolute/path/to/RAVDESS
export PKU_ER_ROOT=/absolute/path/to/PKU-AV
```

Every waveform must be 3.6 seconds long. The acoustic tensor has 820 channels and 156 positions: MFCC 10, CQT 42, and a 768-dimensional frozen embedding.

### RAVDESS, 30 frames

Run the following only when the corresponding preprocessing step has not already been completed. `prepare_raw_layout.py` is needed only when the downloaded MP4 files are stored as flat `Actor_XX` directories.

```bash
python ravdess_preprocessing/prepare_raw_layout.py
python ravdess_preprocessing/extract_audios.py
python ravdess_preprocessing/extract_faces.py
python ravdess_preprocessing/create_annotations.py
python ravdess_preprocessing/extract_faces_30f.py
python ravdess_preprocessing/create_annotations_5fold.py \
  --src ravdess_preprocessing/annotations_00.txt \
  --out ravdess_preprocessing/annotations_fold30f.txt \
  --n_folds 5 \
  --rebase_from Video_Speech_Actors_01-24_processed \
  --rebase_to Video_Speech_Actors_01-24_processed_30f
```

This creates `annotations_fold30f_0.txt` through `annotations_fold30f_4.txt`. The annotation format is:

```text
video_path;audio_path;label;subset
```

`subset` is `training`, `validation`, or `testing`; labels are one-based. The annotation builder appends to an existing output file, so remove or relocate previously generated annotations before rebuilding them.

The five test-actor groups used in the reported RAVDESS evaluation are:

```text
fold 0: 10 11 17 19 23
fold 1: 01 03 08 14 20
fold 2: 02 04 07 12 13
fold 3: 05 06 09 15 22
fold 4: 16 18 21 24
```

The script generates actor-wise folds from its `--seed` setting, but the default random grouping is not asserted to reproduce the IDs above. For exact reproduction of the reported experiments, set `groups` in `create_annotations_5fold.py` to these actor IDs and verify every generated fold before training.

### PKU-ER

The scripts are in `PKU_ER_preprocessing/`. A typical sequence is:

```bash
python PKU_ER_preprocessing/extract_audios.py
python PKU_ER_preprocessing/extract_faces_30.py
python PKU_ER_preprocessing/create_annotations.py
python ravdess_preprocessing/create_annotations_5fold.py \
  --src "$PKU_ER_ROOT/dataset_split.txt" \
  --out PKU_ER_preprocessing/annotations_pku_30f.txt \
  --n_folds 5 --stratified \
  --rebase_from Video_processed_face --rebase_to Video_processed_face_30f
```

PKU-ER uses clip-level, class-stratified folds because per-clip speaker identities are unavailable. This protocol does not establish subject-independent generalization.

### Optional acoustic-feature cache

```bash
python extract_beats_cache.py \
  --annotation ravdess_preprocessing/annotations_fold30f_0.txt,ravdess_preprocessing/annotations_fold30f_1.txt,ravdess_preprocessing/annotations_fold30f_2.txt,ravdess_preprocessing/annotations_fold30f_3.txt,ravdess_preprocessing/annotations_fold30f_4.txt \
  --backbone emotion2vec \
  --out_dir /absolute/path/to/cache_e2v
```

Use a separate cache for each acoustic backbone. Waveform-noise evaluation always recomputes features online and does not use the clean-feature cache.

## Data-free checks

```bash
python -m compileall -q .
python noise_utils.py
python analyze_stats.py --self_test
python main.py --help
python run_abl_ravdess.py --list
python run_abl_ravdess.py --id P1-1 --folds 0 --dry_run
```

A model-only forward check, without data or checkpoints:

```bash
python - <<'PY'
import torch
from models.multimodalcnn import MultiModalCNN
for fusion in ['lt', 'it', 'ia', 'gs_it', 'gs_ia', 'gs_lt']:
    model = MultiModalCNN(num_classes=8, fusion=fusion, seq_length=15,
                          pretr_ef='None', num_heads=1)
    audio = torch.randn(2, 820, 156)
    video = torch.randn(2 * 15, 3, 224, 224)
    with torch.no_grad():
        output = model(audio, video, 0.1)
    print(fusion, {key: tuple(value.shape) for key, value in output.items()})
PY
```

## Training

The main RAVDESS configuration uses 30 frames, emotion2vec+, Variant 1 (`gs_it`), NIT, and actor-wise five-fold validation:

```bash
python main.py \
  --dataset RAVDESS \
  --annotation_path ravdess_preprocessing/annotations_fold30f.txt \
  --n_folds 5 --n_classes 8 --sample_duration 30 \
  --audio_backbone emotion2vec \
  --fusion gs_it --num_heads 1 --har full --cross_dir both \
  --mask noise --batch_size 4 --n_epochs 100 \
  --beats_cache_dir /absolute/path/to/cache_e2v \
  --result_path results/main --summary_csv results/main.csv \
  --n_threads 8 --test_only_both
```

For PKU-ER, use `--dataset PKU-ER --n_classes 3 --fusion gs_ia` and the corresponding annotation file. `--pretrain_path None` disables the optional EfficientFace checkpoint.

## The 80-run ablation batch

Inspect configurations and commands without training:

```bash
python run_abl_ravdess.py --list
python run_abl_ravdess.py --id P1-1 --folds 0 --epochs 2 \
  --result_path results/smoke --summary_csv results/smoke.csv --dry_run
```

Run a two-epoch smoke test by removing `--dry_run`. Run all 16 configurations over five folds with:

```bash
mkdir -p logs
python -u run_abl_all.py 2>&1 | tee logs/abl_master.log
```

`run_abl_all.py` skips a configuration/fold pair when its result is already present in the summary CSV. The defaults produce `results_abl_rav30/` and `abl_rav30.csv`; both are ignored by Git.

The following environment variables customize the ablation runner:

| Variable | Default |
|---|---|
| `DRAV_ANNOTATION` | `ravdess_preprocessing/annotations_fold30f.txt` |
| `DRAV_BEATS_CACHE` | Empty, meaning online feature extraction |
| `DRAV_RESULT_PATH` | `results_abl_rav30` |
| `DRAV_SUMMARY_CSV` | `abl_rav30.csv` |
| `DRAV_DEVICE` | `cuda` |
| `DRAV_THREADS` | `8` |

## Robustness and prediction analysis

After training, pass the experiment's generated `store_name` and the same model configuration to `robustness_test.py`:

```bash
python robustness_test.py \
  --result_path results/main --store_name <store_name> \
  --n_folds 5 --annotation_path ravdess_preprocessing/annotations_fold30f.txt \
  --audio_backbone emotion2vec --sample_duration 30 \
  --fusion gs_it --num_heads 1 --mask noise \
  --snr_list clean,20,15,10,5,0,-5 \
  --dump_dir dumps --snr_summary_csv q3_snr.csv

python analyze_stats.py --dumps 'dumps/*snrclean.csv' --out_csv q9_stats.csv
```

## Reproducibility notes

- The training seed is 1 by default. Waveform noise is seeded by a hash of the noise seed, filename, and SNR.
- The Gumbel gate remains stochastic during evaluation. Separate evaluation scripts do not reset a dedicated gate seed, so repeated fused-head evaluations can differ.
- The implemented temperature index is `10*epoch + batch_index`, not a global mini-batch counter. The temperature can increase at epoch boundaries; it is not globally monotonic.
- The nominal batch size of four clips becomes 12 under NIT and 16 under HM through sample copying. These copies are correlated and are not independent clips.
- The 80-run ablation is a new independent training batch and should not be confused with repeated evaluations of the main five-fold checkpoints.

## License and attribution

This project is distributed under the MIT License; see `LICENSE`. It includes derivative work from [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs), [EfficientFace](https://github.com/zengqunzhao/EfficientFace), and [BEATs](https://github.com/microsoft/unilm/tree/master/beats). Follow the original datasets' and pretrained models' licenses and access conditions when downloading them separately.
