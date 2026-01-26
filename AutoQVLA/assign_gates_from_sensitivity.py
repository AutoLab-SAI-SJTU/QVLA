"""
基于一阶 proxy（proxy_{b}）做全局贪心比特分配，与 mydesk/gatedquantv2/alloc_greedy_first_order.py 保持一致：
- 初始全部通道视为最高 bit（默认 16，代表 fp16 不量化）。
- 每次选择“单位比特节省”代价最小的降级步骤（proxy_b / (cur_bit - b)），直到平均 bit 达到目标。
- proxy_{b} 直接视为把该通道量化到 b-bit 的误差代价。
"""
import argparse
import heapq
import json
import os
from typing import Dict, List, Tuple

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Greedy bit allocation from first-order proxy (AutoQVLA).")
    p.add_argument("--proxy_pt", type=str, required=True, help="first_order_proxy_xxx.pt，需包含 proxy_{b}")
    p.add_argument(
        "--bits",
        type=str,
        default="0,2,4,8,16",
        help="可用比特集合（降序/升序均可），例如 0,2,4,8,16；其中 16 表示 fp16 不量化",
    )
    p.add_argument(
        "--target_avg_bits",
        type=float,
        default=4.0,
        help="目标平均比特数（全局）。",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default="mydesk/gatedquantv2/out/action_mse/greedy_bits_from_first_order.json",
    )
    return p.parse_args()


def load_proxies(pt_path: str, bits: List[int]) -> Dict[str, Dict[int, torch.Tensor]]:
    raw = torch.load(pt_path, map_location="cpu")
    res: Dict[str, Dict[int, torch.Tensor]] = {}
    for layer, v in raw.items():
        if not isinstance(v, dict):
            continue
        layer_dict: Dict[int, torch.Tensor] = {}
        for b in bits:
            key = f"proxy_{b}"
            if key in v:
                layer_dict[b] = v[key].float()
        if layer_dict:
            res[layer] = layer_dict
    return res


def next_lower_bit(b: int, bit_list_desc: List[int]) -> int:
    for bb in bit_list_desc:
        if bb < b:
            return bb
    return -1


def greedy_allocate(
    proxies: Dict[str, Dict[int, torch.Tensor]],
    bit_list: List[int],
    target_avg: float,
) -> Tuple[Dict[str, List[int]], Dict[str, float]]:
    bit_list_desc = sorted(bit_list, reverse=True)
    highest = bit_list_desc[0]
    layer_bits: Dict[str, List[int]] = {}
    total_bits = 0
    n_ch = 0
    heap = []  # (unit_cost, step_id, layer, idx, new_bit)
    step_id = 0

    def push_candidate(layer: str, idx: int, cur_bit: int):
        nb = next_lower_bit(cur_bit, bit_list_desc)
        if nb < 0:
            return
        if nb not in proxies[layer]:
            return
        saving = cur_bit - nb
        if saving <= 0:
            return
        cost = float(proxies[layer][nb][idx])
        unit_cost = cost / saving
        nonlocal step_id
        heapq.heappush(heap, (unit_cost, step_id, layer, idx, nb))
        step_id += 1

    # 初始化
    for layer, v in proxies.items():
        any_tensor = next(iter(v.values()))
        C = any_tensor.numel()
        layer_bits[layer] = [highest] * C
        total_bits += highest * C
        n_ch += C
        for ci in range(C):
            push_candidate(layer, ci, highest)

    avg_bit = total_bits / max(1, n_ch)

    # 单步贪心降级
    while heap and avg_bit > target_avg:
        unit_cost, _, layer, idx, nb = heapq.heappop(heap)
        cur_bit = layer_bits[layer][idx]
        # 可能已有过期项
        if nb >= cur_bit:
            continue
        saving = cur_bit - nb
        if saving <= 0:
            continue
        cost = float(proxies[layer][nb][idx])
        total_bits -= saving
        layer_bits[layer][idx] = nb
        avg_bit = total_bits / n_ch
        # 推入下一档
        push_candidate(layer, idx, nb)

    stats = {
        "target_avg_bits": target_avg,
        "final_avg_bits": avg_bit,
        "total_channels": n_ch,
        "initial_total_bits": highest * n_ch,
        "final_total_bits": total_bits,
        "bit_hist": {},
    }
    hist: Dict[int, int] = {}
    for arr in layer_bits.values():
        for b in arr:
            hist[b] = hist.get(b, 0) + 1
    stats["bit_hist"] = {int(k): int(v) for k, v in sorted(hist.items())}
    return layer_bits, stats


def main():
    args = parse_args()
    bits = sorted({int(x) for x in args.bits.split(",") if x.strip()})
    proxies = load_proxies(args.proxy_pt, bits)
    if not proxies:
        raise ValueError("proxy_pt 中未找到任何 proxy_{b} 数据")

    layer_bits, stats = greedy_allocate(proxies, bits, args.target_avg_bits)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        "proxy_pt": args.proxy_pt,
        "bits": bits,
        "assign": layer_bits,
        "stats": stats,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved to {args.out_json}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

