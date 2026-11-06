# Standalone OpenVLA / OpenVLA-OFT weight-only workflow

Both backends retain the original AutoLab repository layout. Their numerical
implementations live in `openvla/qvla` and `openvla-oft/qvla`. The repository-local
`scripts/run_qvla.py` prepares training records, calibrates, allocates and evaluates
without requiring a separate integration repository.

## Start here

Follow the sections in order: fetch this implementation, install the selected
backend, set up LIBERO, download checkpoints/training data, run a startup check,
then launch the full Spatial evaluation and verify its output. All shell blocks
below run from the repository root unless a working directory is stated.

Clone the `main` branch:

```bash
git clone https://github.com/AutoLab-SAI-SJTU/QVLA.git
cd QVLA
export QVLA_ROOT="$PWD"
export OPENVLA_PYTHON="$QVLA_ROOT/.venv-openvla/bin/python"
export OFT_PYTHON="$QVLA_ROOT/.venv-oft/bin/python"
git rev-parse HEAD
```

Keep the printed source revision and generated manifests with the run. Use the
same shell for the following environment variables and commands.

## Recipes

| Backend | Mean weight bits | Calibration attention | Evaluation attention | Pruning repair |
| --- | --- | --- | --- | --- |
| `openvla` | 4 | Causal SDPA | Causal SDPA | 0 |
| `openvla` | 8 | Causal SDPA | Causal FlashAttention 2.5.5 | 0.25 |
| `openvla-oft` | 4 | Bidirectional SDPA | Bidirectional SDPA | 0 |
| `openvla-oft` | 8 | Bidirectional SDPA | Bidirectional SDPA | 0 |

W4/W8 means average output-channel bits in allocated Linear/Conv2d modules.
It is not a parameter-weighted whole-model storage budget. Candidate bits are
`0,4,8,16` and the per-layer zero-bit cap defaults to 1%. Quantization uses
per-output-channel symmetric RTN with BF16 scale arithmetic. The multimodal
projector, language-model output head and OFT action/proprio heads stay BF16.
Zero-bit weights retain the original bias. Activations and operators are BF16
(A16), so this workflow does not emit packed integer kernels or storage savings.

The global allocator minimizes marginal sensitivity increase per saved bit.
OpenVLA W8 also restores the highest-benefit 25% of pruned channels to 8 bits,
paired with the same number of lowest-cost 16-to-8 demotions to conserve the
budget. This is a local recipe, controlled by `--restore-pruned-fraction`.
OFT uses zero repair. These candidate/pruning/sample choices are explicit local
settings rather than unpublished author configurations.

## Install the selected backend

Use separate Python 3.10 environments. OpenVLA needs stock causal
`transformers==4.40.1`; OFT needs the author's bidirectional fork pinned to
`bc339d9ad707454c0c115970db43c260067c61ab`. They have the same version string,
so functional attention checks guard environment selection.

From the repository root, choose one of the following environments:

```bash
python3.10 -m venv .venv-openvla
source .venv-openvla/bin/activate
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e openvla
python -m pip install -r openvla/qvla/requirements.txt
python -m pip install wheel packaging psutil ninja
python -m pip install flash-attn==2.5.5 --no-build-isolation
```

```bash
python3.10 -m venv .venv-oft
source .venv-oft/bin/activate
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e openvla-oft
python -m pip install -r openvla-oft/qvla/requirements.txt
```

FlashAttention is required only for the default OpenVLA W8 evaluator. An existing
2.5.5 installation overlay can be passed with `--flash-attn-path PATH` or
`OPENVLA_FLASH_ATTN_PATH`. No machine-specific path is embedded in the sources.
The current interpreter is used by default; `--python /path/to/env/bin/python`
selects the matching backend environment without resolving away venv identity.

## Set up LIBERO and headless simulation

Install the system OSMesa/OpenGL libraries and ffmpeg. For Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6 libgl1 ffmpeg
```

Clone the pinned LIBERO source into a locally ignored directory:

```bash
mkdir -p openvla-oft/third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git openvla-oft/third_party/LIBERO
git -C openvla-oft/third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
export LIBERO_ROOT="$QVLA_ROOT/openvla-oft/third_party/LIBERO"
```

Install that same LIBERO source into each backend environment you created:

```bash
"$OPENVLA_PYTHON" -m pip install -e "$LIBERO_ROOT"
"$OFT_PYTHON" -m pip install -e "$LIBERO_ROOT"
```

If you installed only one backend, run only its installation command. The QVLA
requirements pin robosuite 1.4.1 and MuJoCo 2.3.7. Create a private LIBERO config
**before importing LIBERO**; it points to the checkout's assets and fixed default
initial states rather than another project's configuration:

```bash
export LIBERO_CONFIG_PATH="$QVLA_ROOT/out/libero_config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
python3 - <<'PYCONFIG'
import json
import os
from pathlib import Path
source = Path(os.environ["LIBERO_ROOT"]) / "libero/libero"
config = Path(os.environ["LIBERO_CONFIG_PATH"])
config.mkdir(parents=True, exist_ok=True)
datasets = source.parent / "datasets"
datasets.mkdir(exist_ok=True)
paths = {"benchmark_root": source, "bddl_files": source / "bddl_files",
         "init_states": source / "init_files", "assets": source / "assets",
         "datasets": datasets}
for key in ("bddl_files", "init_states", "assets"):
    if not paths[key].is_dir():
        raise RuntimeError(f"Missing LIBERO resource: {paths[key]}")
(config / "config.yaml").write_text(json.dumps({key: str(value) for key, value in paths.items()}, indent=2) + "\n")
PYCONFIG
```

The LIBERO simulation assets/default initial states and RLDS training data serve
different purposes. Calibration uses the training data below; robot evaluation
uses the pinned LIBERO benchmark and its default initial states.

## Download local checkpoints and calibration data

Existing local assets can be selected through `--checkpoint` and `--rlds-root`;
downloads are needed only when you do not already have the matching assets.
Use the Python environment of a backend you installed for the downloads. The
following example uses the causal environment; replace the executable with
`"$OFT_PYTHON"` if you installed only OFT. Download only the model(s) you need:

```bash
export QVLA_DOWNLOAD_PYTHON="$OPENVLA_PYTHON"
"$QVLA_DOWNLOAD_PYTHON" - <<'PYDOWNLOAD'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ["QVLA_ROOT"])
# OpenVLA Spatial. Omit this call when evaluating only OFT.
snapshot_download(
    repo_id="openvla/openvla-7b-finetuned-libero-spatial",
    revision="962318cec55ac10993ff0f5f43eda9a270b4c873",
    local_dir=str(root / "models/openvla_spatial"),
)
# Merged OFT Spatial, including the L1 action head and proprio projector.
# Omit this call when evaluating only causal OpenVLA.
snapshot_download(
    repo_id="moojink/openvla-7b-oft-finetuned-libero-spatial",
    revision="6d0231af0e48c5985f1ff86908f4674b84bc049b",
    local_dir=str(root / "models/oft_spatial"),
    ignore_patterns=["adapter/*"],
)
# Shared training data: all four suites are needed for default calibration.
snapshot_download(
    repo_id="openvla/modified_libero_rlds", repo_type="dataset",
    revision="6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551",
    local_dir=str(root / "data/modified_libero_rlds"),
)
PYDOWNLOAD
```

An adapter-only OFT checkpoint is insufficient. Keep checkpoint metadata,
processor/tokenizer, normalization statistics, action head and proprio projector.
The expected data layout is:

```text
data/modified_libero_rlds/
  libero_spatial_no_noops/1.0.0/dataset_info.json
  libero_object_no_noops/1.0.0/dataset_info.json
  libero_goal_no_noops/1.0.0/dataset_info.json
  libero_10_no_noops/1.0.0/dataset_info.json
  ... TFRecord shards inside each suite directory
```

Default calibration selects 512 distinct training trajectories across four
suites, one frame each. Optional `--calib-indices PATH` fixes a supplied
suite/episode/frame list while rereading images/state from RLDS. Simulator reset
screenshots are not calibration inputs. The input manifest stores image hashes.

OpenVLA uses one front camera with Lanczos resize and a single 90%-area center
crop. Its directional score differentiates a temperature-softmax expectation of
action bins conditioned on generated action prefixes. This is a soft surrogate,
not the derivative of discrete argmax actions. Generation/teacher-forcing token
agreement is recorded rather than assumed exact under BF16 kernels.

OFT uses both raw cameras, 8D proprio, official preprocessing, the L1 head and
native 8x7 continuous actions. Its primary score is the Appendix G
variance-times-Jacobian proxy. The first eight gradient-enabled forwards must
match official inference. The earlier input-Gram estimator is replaced.
Neither route includes unpublished sensitive-channel forward refinement or
rollout cross-validation. The prompt token and attention mask are extended
together in OpenVLA. Earlier rollout scores must be revalidated after this fix.

## Check startup before full evaluation

Use a fresh output root for a startup check with eight calibration samples and
one trial per task. It exercises the complete preparation/calibration/allocation/
evaluation path; its ten episodes are not the full Spatial protocol:

```bash
CUDA_VISIBLE_DEVICES=0 "$OFT_PYTHON" scripts/run_qvla.py \
  --backend openvla-oft --checkpoint "$QVLA_ROOT/models/oft_spatial" \
  --rlds-root "$QVLA_ROOT/data/modified_libero_rlds" \
  --task_suite_name libero_spatial --target-avg-bits 8 \
  --max-samples 8 --trials 1 \
  --output-root "$QVLA_ROOT/out/oft_startup" --action run
```

For causal OpenVLA, use its Python, backend and checkpoint. This is optional and
has a separate output directory; full evaluation below recalibrates 512 samples.
If you reuse this example, select a fresh directory instead of overwriting a run.

## Run the four Spatial weight-only configurations

Choose one visible GPU. Run the configuration(s) you need; the commands below
are sequential, with a separate output directory for each. They fix the suite,
512 training trajectories, 50 trials per task and an automatically generated run
seed. The launcher selects
the attention and pruning-repair recipe from the table automatically.

**OpenVLA W4A16:**

```bash
CUDA_VISIBLE_DEVICES=0 "$OPENVLA_PYTHON" scripts/run_qvla.py \
  --backend openvla --checkpoint "$QVLA_ROOT/models/openvla_spatial" \
  --rlds-root "$QVLA_ROOT/data/modified_libero_rlds" \
  --task_suite_name libero_spatial --target-avg-bits 4 \
  --max-samples 512 --trials 50 \
  --output-root "$QVLA_ROOT/out/openvla_w4" --action run
```

**OpenVLA W8A16:**

```bash
CUDA_VISIBLE_DEVICES=0 "$OPENVLA_PYTHON" scripts/run_qvla.py \
  --backend openvla --checkpoint "$QVLA_ROOT/models/openvla_spatial" \
  --rlds-root "$QVLA_ROOT/data/modified_libero_rlds" \
  --task_suite_name libero_spatial --target-avg-bits 8 \
  --max-samples 512 --trials 50 \
  --output-root "$QVLA_ROOT/out/openvla_w8" --action run
```

**OpenVLA-OFT W4A16:**

```bash
CUDA_VISIBLE_DEVICES=0 "$OFT_PYTHON" scripts/run_qvla.py \
  --backend openvla-oft --checkpoint "$QVLA_ROOT/models/oft_spatial" \
  --rlds-root "$QVLA_ROOT/data/modified_libero_rlds" \
  --task_suite_name libero_spatial --target-avg-bits 4 \
  --max-samples 512 --trials 50 \
  --output-root "$QVLA_ROOT/out/oft_w4" --action run
```

**OpenVLA-OFT W8A16:**

```bash
CUDA_VISIBLE_DEVICES=0 "$OFT_PYTHON" scripts/run_qvla.py \
  --backend openvla-oft --checkpoint "$QVLA_ROOT/models/oft_spatial" \
  --rlds-root "$QVLA_ROOT/data/modified_libero_rlds" \
  --task_suite_name libero_spatial --target-avg-bits 8 \
  --max-samples 512 --trials 50 \
  --output-root "$QVLA_ROOT/out/oft_w8" --action run
```

`--trials 1` gives 10 episodes, `--trials 10` gives 100, and `--trials 50` gives
500. Only the last is the full Spatial protocol documented here. Do not compare
success rates from different episode counts as the same experiment.

## Verify the result

For an OFT W8 run:

```bash
"$OFT_PYTHON" scripts/run_qvla.py --backend openvla-oft \
  --action result --output-root "$QVLA_ROOT/out/oft_w8"
"$OFT_PYTHON" openvla-oft/qvla/verify_eval.py \
  --run_dir "$QVLA_ROOT/out/oft_w8/logs/eval"
```

Use the corresponding backend verifier and output directory for the other three
configurations. A completed full Spatial run must have 500 episodes and matching
canonical-log/metadata counters. Read `total_success_rate` in
`merged_summary.json` (multiply by 100 for percent); independent verification
must report `status: passed`. Missing/incomplete summaries or runtime exceptions
are not valid SR measurements. Per-run videos/logs stay under `logs/eval/`.
No expected SR numbers or local experiment outputs are bundled with this update.

## Staged runs and exact input selection

A fresh output root gets a random 32-bit seed; there is no fixed default. The
runner saves it in `.qvla_seed.json` and passes it to calibration sampling,
action projections and evaluation. Separate invocations using the same output
root reuse it, so stage caches and run metadata stay consistent. New output
roots generate new seeds. Optional `--seed N` chooses an explicit replay seed;
changing an existing run's seed requires a new output root. `plan` is read-only
and reports a null seed until one is provided or recorded. Direct backend
commands also generate and record a seed when omitted, per invocation.

`--action plan` prints the resolved recipe without loading a model or creating an
output root. Keep your selected command's other arguments and replace `run` with
`build_calib`, `proxy`, `assign`, `eval` or `export` to execute one stage. For a
complete staged run, execute preparation, proxy, allocation and evaluation in that
order with the same configuration and output root. `result` reads the summary.
An independent evaluation can use `--action eval --gates_path PATH` with a new
output root. `--attention` and `--calibration-attention` select explicit attention;
OFT requires bidirectional SDPA. The old input-Hessian `--max_layers`, `--resume`
and `--percdamp` flags are rejected.

The default sampler uses each run's generated seed for training selection. To
reproduce another run's exact frame list, add `--calib-indices /path/to/indices.jsonl`
and set `--max-samples` to that list's length before preparation. Required fields
are `suite`, `episode_index` and `step_index`; an optional instruction is checked.
Keep the resulting manifest, indices and source/environment/checkpoint identities
with your run. The repository does not ship private local calibration indices or
unpublished author indices. These steps reproduce the released weight-only
workflow; they do not claim to reproduce every paper result or omitted refinement
stage. Direct-file/module APIs are described in the backend guides.

## Artifacts and checks

Each stage records a `.qvla_<stage>.json` manifest bound to checkpoint/source/
environment/input/configuration identities. Stale, partial and untracked files
are rejected. Outputs include `calib/records.jsonl`, `proxy/proxy.pt`,
`gates/greedy_bits.json` and per-run metadata/logs/videos under `logs/eval/`.
`merged_summary.json` and `.md` are written only after independent canonical-log
verification succeeds. Runtime errors abort rather than becoming policy failures.
Protected components and source identities are checked during runs. Bad gate
names/lengths/bits and pruning-cap violations fail before any weight mutation.

Optional export saves another full BF16 checkpoint with injected fake weights,
not a packed low-bit model. `--method bf16` selects floating-point evaluation
with the same guarded loader. Default process memory fractions are 0.55 for
calibration and 0.42 for evaluation; configure the corresponding CLI options
for a smaller dedicated GPU if necessary.

```bash
python -m pytest tests -q
cd openvla
CUDA_VISIBLE_DEVICES='' python -m pytest qvla/tests -q
# Run the OFT backend tests separately from openvla-oft/ in its own environment.
```

CI uses separate causal and bidirectional environments. Tests cover action
sensitivity, BF16 quantization, marginal allocation, pruning budgets, source/data
contracts, strict log auditing and standalone stage/cache behavior.
