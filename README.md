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

QVLA provides a quantization workflow for VLA models, including
proxy sensitivity estimation, greedy gate assignment, and quantization for evaluation or checkpoint export.


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

- Channel-wise gates over `{0,2,4,8,16}` with a target global average bit-width.
- Works on `language_model.*` and `vision_backbone.*` Linear/Conv2d layers.
- Excludes `projector.*`, `action_head`, and `language_model.lm_head`.

## 🖼️ Figure
Sensitivity Analysis
![Sensitivity Analysis](assets/motivation.png)

## 📁 Repository Layout

- `openvla/` OpenVLA source code and dependencies
- `openvla/qvla/` QVLA workflow scripts

## ⚙️ Installation

The examples below use OpenVLA as the backend.

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Update CUDA version to match your system.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME

# Install OpenVLA in editable mode
pip install -e openvla

# Minimal dependencies for QVLA scripts
pip install -r openvla/requirements-min.txt
pip install pillow tqdm

# (Optional) Flash Attention 2 for training
pip install packaging ninja
ninja --version; echo $?  # should return exit code 0
pip install "flash-attn==2.5.5" --no-build-isolation
```

## 🚀 Usage

All commands below are run from the repository root.

### 1) Build proxy sensitivity

```bash
python openvla/qvla/sensitivity_hessian_proxy.py \
  --pretrained_checkpoint path/to/openvla_checkpoint \
  --calib_jsonl path/to/calib.jsonl \
  --out_path out/proxy.pt \
  --bits 0,2,4,8
```

### 2) Assign gates

```bash
python openvla/qvla/assign_gates_from_sensitivity.py \
  --proxy_pt out/proxy.pt \
  --bits 0,4,8,16 \
  --target_avg_bits 8.0 \
  --out_json out/greedy_bits.json
```

### 3) Inject weight-only fake quant and save model

```bash
python openvla/qvla/inject_fake_w.py \
  --pretrained_checkpoint path/to/openvla_checkpoint \
  --gates_path out/greedy_bits.json \
  --out_dir out/openvla_qvla_fakew
```

### 4) Evaluate on LIBERO

Set `LIBERO_ROOT` to your local LIBERO checkout if needed.

```bash
export LIBERO_ROOT=path/to/LIBERO

python openvla/qvla/run_eval_with_qvla_fakew.py \
  --pretrained_checkpoint path/to/openvla_checkpoint \
  --gates_path out/greedy_bits.json \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --local_log_dir out/rollouts_qvla
```

## 📦 Outputs

- `out/proxy.pt`: per-layer `proxy_{b}` tensors
- `out/greedy_bits.json`: channel-wise gate assignment
- `out/openvla_qvla_fakew/`: exported fake-quantized checkpoint
- `out/rollouts_qvla/`: evaluation logs


## 📚 Citation

```bibtex
@inproceedings{xu2026qvla,
  title     = {QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization},
  author    = {Xu, Yuhao and Yang, Yantai and Fan, Zhenyang and Liu, Yufan and Li, Yuming and Li, Bing and Zhang, Zhipeng},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
