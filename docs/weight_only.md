# Standalone OpenVLA / OpenVLA-OFT weight-only workflow

Both backends retain the original AutoLab repository layout. Their numerical
implementations live in `openvla/qvla` and `openvla-oft/qvla`. The repository-local
`scripts/run_qvla.py` prepares training records, calibrates, allocates and evaluates
without requiring a separate integration repository.

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

Install LIBERO at `8f1084e3132a39270c3a13ebe37270a43ece2a01`, robosuite 1.4.1,
MuJoCo 2.3.7, ffmpeg and the system OSMesa library. Set `LIBERO_ROOT`, a private
`LIBERO_CONFIG_PATH`, `MUJOCO_GL=osmesa`, and `PYOPENGL_PLATFORM=osmesa`.
The [OFT setup guide](../openvla-oft/qvla/README.md#installation) includes LIBERO
asset and private-config instructions; keep each backend's own Python dependencies.

## Local checkpoints and calibration data

Download the merged backend checkpoint and `openvla/modified_libero_rlds` from
Hugging Face. The RLDS root contains
`libero_{spatial,object,goal,10}_no_noops/1.0.0/` training datasets. Use
`openvla/openvla-7b-finetuned-libero-spatial` for OpenVLA, or
`moojink/openvla-7b-oft-finetuned-libero-spatial` for OFT. OFT additionally needs
its continuous action head and proprio projector; adapter-only weights are
insufficient. The OFT guide pins the checkpoint revision used by this route.

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

## Run W4A16 or W8A16

From the repository root, choose one visible GPU and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_qvla.py \
  --backend openvla --python /path/to/causal-env/bin/python \
  --checkpoint /path/to/openvla-7b-finetuned-libero-spatial \
  --rlds-root /path/to/modified_libero_rlds \
  --target-avg-bits 8 --max-samples 512 --trials 50 --seed 7 \
  --output-root out/openvla_w8 --action run

CUDA_VISIBLE_DEVICES=0 python scripts/run_qvla.py \
  --backend openvla-oft --python /path/to/oft-env/bin/python \
  --checkpoint /path/to/openvla-7b-oft-finetuned-libero-spatial \
  --rlds-root /path/to/modified_libero_rlds \
  --target-avg-bits 8 --max-samples 512 --trials 50 --seed 7 \
  --output-root out/oft_w8 --action run
```

Use `--target-avg-bits 4` and a new output for W4A16. `--trials 1` is a ten-task
startup check; `--trials 10` runs 100 episodes; `--trials 50` runs 500 episodes.
These protocols must be reported separately. No new experimental outcomes are
bundled with this code release.

`--action plan` prints the resolved recipe without loading models or writing an
output directory. Individual `build_calib`, `proxy`, `assign`, `eval`, `export`
and `result` actions use the same arguments/output root. For a startup check,
use `--max-samples 8 --trials 1` from data preparation onward. An independent
evaluation can use `--action eval --gates_path PATH` with a fresh output root.
`--attention` and `--calibration-attention` explicitly select attention;
OFT requires bidirectional SDPA. The old input-Hessian `--max_layers`, `--resume`
and `--percdamp` flags are rejected. Direct-file/module APIs remain available;
see the backend guides.

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
