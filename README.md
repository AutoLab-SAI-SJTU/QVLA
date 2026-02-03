# QVLA说明

## 运行步骤（建议流程）

下面命令假定你在仓库根目录执行，并用占位路径替换实际位置：

```
cd /path/to/openvla
```

### 1) 准备校准数据（JSONL）

每行包含 `text` 与 `image` 字段：

```
{"text": "pick up the cube", "image": "/path/to/image1.jpg"}
{"text": "open the drawer", "image": "/path/to/image2.jpg"}
```

### 2) 计算 Hessian proxy

该步骤基于一阶 Hessian 统计生成通道敏感度，输出包含 `proxy_{b}` 的文件，供后续 gates 分配使用。

实现参考（Hessian 统计与权重量化逻辑）：

- `openvla/gptq/gptq.py` 的 `GPTQ.add_batch()` / `H` 累积
- `AutoQVLAv2/Activation/QuaRot/fake_quant/gptq_utils.py` 的 `GPTQ` 实现

请使用你的 Hessian‑proxy 生成脚本，产出如下格式的文件：

```
proxy_out.pt  # {layer_name: {"proxy_0": Tensor[C], "proxy_2": Tensor[C], "proxy_4": Tensor[C], "proxy_8": Tensor[C]}}
```

> 如果你已有可用的 proxy（格式包含 `proxy_{b}`），可跳过此步。

### 3) 从 proxy 生成 gates

```
python AutoQVLAv2/assign_gates_from_sensitivity.py \
  --proxy_pt /path/to/proxy_out.pt \
  --target_avg_bits 4.0 \
  --out_json /path/to/gates_with_stats.json
```

`gates_with_stats.json` 包含多个字段，其中真正的 gates 映射在 `assign` 键下，需要提取成独立文件：

```
python - <<'PY'
import json

src = "/path/to/gates_with_stats.json"
dst = "/path/to/gates_map.json"

with open(src, "r") as f:
    data = json.load(f)

with open(dst, "w") as f:
    json.dump(data["assign"], f)

print("saved:", dst)
PY
```

### 4) 注入量化并保存模型

```
python AutoQVLAv2/inject_fake_w.py \
  --pretrained_checkpoint /path/to/openvla_checkpoint \
  --gates_path /path/to/gates_map.json \
  --out_dir /path/to/openvla_autoqvla_fakew \
  --device cuda \
  --dtype bf16
```

### 5) 评测（示例）

使用 OpenVLA 的标准评测脚本，指向上一步保存的模型目录：

```
python /path/to/openvla/experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /path/to/openvla_autoqvla_fakew \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1
```

> 若仓库内存在多个 `experiments` 目录，请选择你当前可用的 OpenVLA 评测入口。

