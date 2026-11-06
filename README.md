# QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization

<div align="center">
    <p>
        <a>Yuhao Xu</a><sup>1</sup>&nbsp;&nbsp;
        <a>Yantai Yang</a><sup>1,2</sup>&nbsp;&nbsp;
        <a>Zhenyang Fan</a><sup>1</sup>&nbsp;&nbsp;
        <a>Yufan Liu</a><sup>3,4</sup>&nbsp;&nbsp;
        <br>
        <a>Yuming Li</a><sup>5</sup>&nbsp;&nbsp;
        <a>Bing Li</a><sup>3</sup>&nbsp;&nbsp;
        <a>Zhipeng Zhang</a><sup>1</sup>&nbsp;&nbsp;
    </p>
    <p>
        <sup>1</sup>AutoLab, School of Artificial Intelligence, Shanghai Jiao Tong University&nbsp;&nbsp;
        <br>
        <sup>2</sup>Anyverse Dynamics&nbsp;&nbsp;
        <br>
        <sup>3</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences&nbsp;&nbsp;
        <br>
        <sup>4</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences&nbsp;&nbsp;
        <br>
        <sup>5</sup>Terminal Technology Department, Alipay, Ant Group
    </p>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2602.03782">
    <img src="https://img.shields.io/badge/arXiv-2602.03782-b31b1b.svg" alt="arXiv">
  </a>
</p>




QVLA provides action-sensitivity calibration, channel-wise bit allocation, and
quantization for evaluation or checkpoint export. The standalone OpenVLA and
OpenVLA-OFT W4A16/W8A16 workflow is documented in
[the weight-only guide](docs/weight_only.md).

## 📰 News

- [Jan 26, 2026] Accepted to ICLR 2026.

## 📖 Abstract

The advent of Vision-Language-Action (VLA) models represents a significant leap
for embodied intelligence, yet their immense computational demands critically
hinder deployment on resource-constrained robotic platforms. Intuitively, low-bit
quantization is a prevalent and preferred technique for large-scale model
compression. However, we find that a systematic analysis of VLA model's
quantization is fundamentally lacking. We argue that naively applying uniform-bit
quantization from Large Language Models (LLMs) to robotics is flawed, as these
methods prioritize passive data fidelity while ignoring how minor action
deviations compound into catastrophic task failures. To bridge this gap, we
introduce QVLA, the first action-centric quantization framework specifically
designed for embodied control. In a sharp departure from the rigid, uniform-bit
quantization of LLM-based methods, QVLA introduces a highly granular, channel-wise
bit allocation strategy. Its core mechanism is to directly measure the final
action-space sensitivity when quantizing each individual channel to various
bit-widths. This process yields a precise, per-channel importance metric that
guides a global optimization, which elegantly unifies quantization and pruning
(0-bit) into a single, cohesive framework. Extensive evaluations on different
baselines demonstrate the superiority of our approach. In LIBERO, the quantized
OpenVLA-OFT with QVLA requires only 29.2% of the original model's VRAM while
maintaining 98.9% of its original performance and achieving a 1.49x speedup.
This translates to a 22.6% performance improvement over SmoothQuant.

## ✨ Highlights

- OpenVLA/OpenVLA-OFT weight-only routes allocate `{0,4,8,16}` output-channel bits with a default 1% per-layer pruning cap.
- Works on `language_model.*` and `vision_backbone.*` Linear/Conv2d layers.
- Excludes `projector.*`, `action_head`, and `language_model.lm_head`.

## 🖼️ Figure
Sensitivity Analysis
![Sensitivity Analysis](assets/motivation.png)

## 📁 Repository Layout

- `openvla/` OpenVLA source code and dependencies
- `openvla/qvla/` Causal soft-action calibration, allocation, guarded evaluation and CPU checks
- `openvla-oft/` OpenVLA-OFT source and pinned bidirectional inference dependencies
- `openvla-oft/qvla/` Native continuous-action calibration, allocation, guarded evaluation and CPU checks
- `scripts/run_qvla.py` Standalone stage runner for both weight-only backends
- `docs/weight_only.md` Setup, method scope, recipes and artifact contracts
- `UniVLA/` UniVLA source and its backend-specific QVLA scripts

## ⚙️ Installation

OpenVLA uses stock causal Transformers 4.40.1; OpenVLA-OFT uses the author's
pinned bidirectional fork. Install each backend in its own Python 3.10
environment. See the [setup and method guide](docs/weight_only.md).

## 🚀 Weight-only usage

Run from the repository root after preparing a matching merged checkpoint,
Hugging Face modified LIBERO RLDS training data, and a private LIBERO config:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_qvla.py \
  --backend openvla --python /path/to/causal-env/bin/python \
  --checkpoint /path/to/openvla_checkpoint \
  --rlds-root /path/to/modified_libero_rlds \
  --target-avg-bits 8 --max-samples 512 --trials 50 --seed 7 \
  --output-root out/openvla_w8 --action run
```

For OFT select `--backend openvla-oft`, its own Python environment and merged
checkpoint. Use target 4 for W4A16. Both routes use a shared standalone stage
interface; their numerical implementations and environment checks are separate.
OpenVLA W8 evaluates with causal FlashAttention 2.5.5; W4 uses causal SDPA.
OFT uses bidirectional SDPA for both precisions. Activations and operators remain
BF16, so these fake-weight paths do not provide packed integer speedups.

The default 512 calibration records come from distinct training trajectories
across four LIBERO suites. OpenVLA uses a soft-action directional surrogate;
OFT differentiates its native continuous action chunk. Local recipe choices and
unreproduced paper stages are documented explicitly in the guide.

`--action plan|build_calib|proxy|assign|eval|export|run|result` selects a stage.
`--trials 1` is a startup check, 10 is a 100-episode protocol, and 50 is the full
500-episode Spatial protocol. Use a fresh output directory for each run.

## 📦 Outputs and validation

- Calibration records, sensitivity tensors and channel assignments live under
  `calib/`, `proxy/` and `gates/` in the selected output root.
- Verified evaluation writes `merged_summary.json`/`.md`, with canonical logs,
  metadata and videos under `logs/eval/`. Runtime faults abort evaluation.
- Stage manifests reject stale or unverified artifacts. Gate names, lengths,
  bit choices and pruning caps are checked before weight mutation.
- Optional export writes a full BF16 checkpoint with injected fake weights.
- Separate CPU test environments cover both backends and the standalone runner.

This update includes code and configuration only. Current experiment results,
weights, datasets, calibration files, scores, gates, logs and videos are not
bundled. The headline paper results above retain their original paper context.

## 📚 Citation

```bibtex
@misc{xu2026qvlachannelsequalvisionlanguageaction,
      title={QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization}, 
      author={Yuhao Xu and Yantai Yang and Zhenyang Fan and Yufan Liu and Yuming Li and Bing Li and Zhipeng Zhang},
      year={2026},
      eprint={2602.03782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03782}, 
}
```
