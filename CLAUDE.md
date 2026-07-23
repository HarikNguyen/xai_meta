# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a research repo backing a thesis on explainable AI (XAI) for gradient-based meta-learning. The
active codebase lives in `src/`: a PyTorch reimplementation of MAML (Model-Agnostic Meta-Learning) plus
a novel post-hoc explainer for it, called **FAMA** (Feature-space Adjoint Meta-learning Attribution).
Everything else in the repo is supporting material, not code to modify unless specifically asked:

- `thesis/` — the LaTeX thesis document itself (compiled via `thesis/Makefile`, which shells out to a
  `latex-small` Docker image running `pdflatex`/`biber`/`makeglossaries`). Chapters map 1:1 to the
  research story implemented in `src/`.
- `cfinn_maml/` — a vendored copy of the original Finn et al. (2017) MAML TensorFlow repo, kept only as
  reference material. It is not wired up to `src/` in any way — don't conflate the two.
- `surveys/` — background-reading PDFs (batch norm, meta-learning, saliency/CAM methods, XAI surveys).
  Reference only.
- `push.sh` / `pull.sh` — rsync a remote GPU workspace over ssh (IP/port cached in a gitignored `.env`).
  Not part of the build; only relevant if asked to sync code to/from a training box.

## Environment setup

```bash
./setup_env.sh   # installs Miniconda, creates conda env "py13" (Python 3.13), installs torch/torchvision
                  # (auto-detects CUDA via nvidia-smi and picks the matching PyTorch index URL, else CPU wheels)
```

There is no `requirements.txt` / `pyproject.toml`. Runtime deps observed in code: `torch`, `torchvision`,
`pyyaml`, `tqdm`, `pandas`, `numpy`, `Pillow`. `torch.func` (functorch-style `vmap`/`grad_and_value`) is
used directly, so a reasonably recent torch is required.

## Dataset preparation

Datasets are plain folders expected in the current working directory (not committed to git). Fetch and
preprocess before running anything:

```bash
cd src
./download_miniimagenet.sh   # -> miniImagenet/
./download_tiered.sh         # -> tiered_imagenet/
./download_cub200.sh         # -> cub_200/  (used for cross-domain / OOD experiments)
python prepare_dataset.py    # resize raw images to 84x84 under train/val/test splits
python prepare_cub.py
python prepare_strokes_omniglot.py
```

## Running experiments

All commands run from `src/`, driven by a YAML config plus a `--mode`:

```bash
cd src
python main.py --config configs/tiered.yaml --mode train --vmap_chunk_size 4
python main.py --config configs/tiered.yaml --checkpoint_dir MAML_tiered_tiered --log_dir tiered --mode test --use_last
python main.py --config configs/tiered.yaml --checkpoint_dir MAML_tiered_tiered --log_dir tiered_ex --mode explain --use_last [--flip_ratio 0.75] [--blur]
python main.py --config configs/explain_test.yaml --checkpoint_dir MAML_tiered_tiered --mode check_explain --use_last --check_method biADT|sanity_params|sanity_support_set
```

`run.sh` documents the canonical set of invocations across all experiment configs (mini2cub, tiered2cub,
tiered, tiered2strokes_omnig) — use it as a reference for argument combinations rather than running it
wholesale (it assumes checkpoints already exist).

Checkpoints are written under `<checkpoint_dir>/<AlgoClassName>/{best,last}_checkpoint.pt`; logs/plots
under `<log_dir>/`. Both `.checkpoints/` and `.logs/` at the repo root are gitignored working directories
used by convention.

## Tests

Tests are `unittest`-based (no pytest config, no conftest) and require real dataset folders (e.g.
`miniImagenet/`) to exist in the CWD — there is no mocked/synthetic fixture data.

```bash
cd src
python -m unittest tests.test_loaders
python -m unittest tests.test_loaders.TestLoaderMiniImageNet.test_sampler_logic  # single test
```

## Architecture

**Config-driven entry point.** `src/main.py` parses CLI args (`--mode`, `--algo`, `--config`,
`--checkpoint_dir`, `--log_dir`, `--use_best`/`--use_last`, explain-specific `--flip_ratio`/`--blur`,
check-explain's `--check_method`) and loads a YAML config with three sections — `dataset`, `dataloader`,
`algo` (see `src/configs/*.yaml`). It hands off to `run()` in `src/scripts/__init__.py`, which dispatches
by mode to `run_train` / `run_test` / `explain` / `check_explain`.

**Warm-up builds everything from config.** `src/scripts/warm_up.py` constructs the train/val/test
dataloaders (and, if `explain_root`/`ood_explain_root` are set in the config, dedicated explain/OOD-explain
loaders for cross-domain explainability experiments), picks the loss (`CrossEntropyLoss` vs
`SmoothMarginLoss` via `algo.criterion`), and assembles `algo_conf` (backbone constructor, optimizer,
device, batch sizes) consumed by the algorithm class.

**Meta-learning loop is fully vectorized across tasks via `torch.func`.** `src/algos/maml.py`'s `MAML`
(subclass of `src/algos/base.py::BaseAlgorithm`) does *not* loop over tasks in a meta-batch in Python.
`_deploy` runs one task's inner-loop adaptation functionally (params passed as an explicit list of
tensors, never mutated on the `nn.Module`), and `train`/`_validate` wrap it in `torch.func.vmap` over the
task dimension with `torch.func.grad_and_value` for the inner-loop gradient. This is why the model
(`src/models/conv4.py::Conv4`) has two forward paths: `forward(x)` for normal use and a functional
`forward(x, weights, only_features=...)` that indexes into a flat weights list per conv block — required
for vmap/grad to work over per-task fast weights. `forward_features` further splits body (conv features)
from head (final linear layer), which the explainer depends on.

**Data pipeline groups by task, not by example.** `src/loaders/datasets.py::FewShotDataset` +
`src/loaders/samplers.py::BatchTaskSampler` sample N-way/K-shot/K-query episodes; `_task_collate` (in
`src/loaders/__init__.py`) collates each task's support/query sets *separately* and one-hot encodes
labels per task, producing a "batch of Tasks" (`boT`) — a list of `(support, query)` tuples rather than
one stacked tensor. `src/loaders/utils.py::boT_to_stack` converts that structure into the stacked
`sup_x, sup_y, que_x, que_y` tensors that `vmap` expects (task dimension first).

**FAMA is the thesis's core contribution.** `src/interpreters/fama.py::FAMAExplainer` is a post-hoc
explainer for a trained MAML model. Given a task's support/query sets, it:
1. Computes two inner-loop adaptation trajectories from the same θ₀: full adaptation `φ_T` (body+head
   both adapt), and `φ_freeze_T` where the body is pinned at θ₀ and only the head evolves along its own
   trajectory (`_get_head_mask` identifies head params via `models/utils.py::get_layer_parameters_map`,
   assuming the last parametrized module is the classifier head).
2. Defines adaptation gain `ΔM = E_Q[L(φ_freeze_T,Q) − L(φ_T,Q)]` — how much adaptation actually helped,
   relative to only tuning the head.
3. Computes `∂ΔM/∂S` (a saliency map over the support set) via an adjoint/Pearlmutter Hessian-vector-product
   backward recursion through each trajectory, then Grad-CAM-style upsamples it to input resolution and
   subtracts the two trajectories' saliencies.

The math and variable names (`φ`, `λ^(t)`, HVP) are spelled out in the module docstring — read it before
touching this file, since the code is a direct transcription of the derivation.

**Evaluating the explainer itself.** `src/scripts/check_explain/` holds faithfulness/sanity checks run
against a trained checkpoint + explainer, selected via `--check_method`:
- `bi_adt.py` — bidirectional faithfulness (perturb by most/least salient regions and measure accuracy
  drop; produces PDAS/NDAS/Combined scores).
- `sanity_params.py` — sanity checks against explanation-invariance-to-parameter tests.
- `sanity_support_set.py` — checks explanation behavior on noisy-label, "hard", and OOD support sets
  (requires both an in-domain and an OOD test loader).

**Cross-domain configs.** Several YAML configs (`mini2cub.yaml`, `tiered2cub.yaml`,
`tiered2strokes_omnig.yaml`) intentionally train on one dataset and test/explain on a different one, to
study how explanations behave under domain shift — this is a deliberate experimental design, not
misconfiguration.
