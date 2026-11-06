# Weight-only QVLA on causal OpenVLA

Use the [standalone workflow guide](../../docs/weight_only.md) for setup,
W4A16/W8A16 recipes and the repository-level stage runner. This backend requires
stock causal Transformers 4.40.1 in its own Python 3.10 environment. Default W8
evaluation additionally uses FlashAttention 2.5.5; W4 uses causal SDPA.

Calibration scores the softmax expectation of action bins conditioned on
generated prefixes. It is a directional soft-action surrogate, not a derivative
of discrete argmax actions. Scores use BF16 quantization perturbations and FP32
accumulation. Activations, protected projector/output-head weights and actual
operators remain BF16. Packed integer inference is outside this workflow.

Direct-file commands from the repository root are also supported:

```bash
python openvla/qvla/prepare_calibration.py \
  --data_root /path/to/modified_libero_rlds \
  --out_dir out/calib --trajectories 512

CUDA_VISIBLE_DEVICES=0 python openvla/qvla/calibrate_actions.py \
  --pretrained_checkpoint /path/to/merged_checkpoint \
  --calib_jsonl out/calib/records.jsonl --out_dir out/proxy

python openvla/qvla/assign_gates_from_sensitivity.py \
  --proxy_pt out/proxy/proxy.pt --bits 0,4,8,16 --target_avg_bits 8 \
  --max_prune_ratio 0.01 --restore_pruned_fraction 0.25 \
  --out_json out/gates_w8.json

CUDA_VISIBLE_DEVICES=0 python openvla/qvla/run_eval.py \
  --pretrained_checkpoint /path/to/merged_checkpoint \
  --gates_path out/gates_w8.json --attention flash_attention_2 \
  --task_suite_name libero_spatial --num_trials_per_task 50 \
  --local_log_dir out/eval_w8

python openvla/qvla/verify_eval.py --run_dir out/eval_w8
```

For W4 allocation use target 4 and repair fraction 0; evaluate with SDPA.
Direct module APIs default to no pruning repair and SDPA; the root stage runner
selects the tabled W4/W8 recipe automatically. Prepared image manifests include
`preprocessed=true`, prompts and SHA256 hashes, preventing duplicate center crop.
The deprecated `sensitivity_hessian_proxy.py` filename redirects to
`calibrate_actions` and expects its new arguments, including `--out_dir`.

The evaluator retains the official LIBERO action path with prompt/mask alignment,
checks protected weights and sources, fails on runtime errors and keeps canonical
logs, metadata and videos per run. Reusing earlier SR numbers requires independent
revalidation. This release includes code/configuration only.
