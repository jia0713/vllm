# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check teaching shapes, slots and explicitly specified ring payload formulas."""

import argparse
import json
from collections import Counter
from pathlib import Path


def ring_bytes(kind, full_bytes, ranks):
    """Return data-only bytes for an equal-sized ring, not measured traffic."""
    assert ranks >= 1 and full_bytes % ranks == 0
    factor = 2 if kind == "all_reduce" else 1
    sent = factor * (ranks - 1) * full_bytes // ranks
    return {
        "algorithm": "equal-sized ring",
        "logical_full_payload_B": full_bytes,
        "per_rank_send_B": sent,
        "per_rank_recv_B": sent,
        "group_send_B": ranks * sent,
        "group_recv_B": ranks * sent,
        "group_send_plus_recv_B": 2 * ranks * sent,
        "per_rank_send_MB": sent / 10**6,
        "per_rank_send_MiB": sent / 2**20,
        "scope": "activation data only; excludes scales and metadata",
    }


def slot(position, block_size, table):
    return table[position // block_size] * block_size + position % block_size


def attention_shapes(config, tp, tokens):
    required = [
        "hidden_size", "num_attention_heads", "head_dim",
        "q_lora_rank", "o_lora_rank", "o_groups",
    ]
    assert all(key in config for key in required)
    hidden, heads, dim, q_rank, o_rank, groups = [
        config[key] for key in required
    ]
    assert heads % tp == 0 and groups % tp == 0 and heads % groups == 0
    return {
        "tp": tp,
        "tokens": tokens,
        "input": [tokens, hidden],
        "fused_wqa_wkv_weight": [q_rank + dim, hidden],
        "wq_b_weight": [heads * dim // tp, q_rank],
        "semantic_q": [tokens, heads // tp, dim],
        "wo_a_grouped_weight": [groups // tp, o_rank, heads * dim // groups],
        "wo_b_weight": [hidden, groups * o_rank // tp],
        "partial_output": [tokens, hidden],
        "shared_kv": [tokens, dim],
        "scope": "semantic dimensions, not packed storage or backend legality",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = {
        "hidden_size": 16, "num_attention_heads": 8, "head_dim": 4,
        "q_lora_rank": 6, "o_lora_rank": 3, "o_groups": 4,
    }
    config_origin = "teaching-only dimensions, not a checkpoint"
    if args.config:
        raw = json.loads(args.config.read_text())
        config = raw.get("text_config", raw)
        config_origin = str(args.config)
    shapes = attention_shapes(config, args.tp, args.tokens)
    ordinary = [slot(p, 4, [7, 2, 9]) for p in [0, 3, 4, 7, 8]]
    assert ordinary == [28, 31, 8, 11, 36]
    compressed = {
        p: slot(p // 4, 4, [5, 8]) if (p + 1) % 4 == 0 else -1
        for p in [3, 4, 7, 11, 15, 19]
    }
    assert list(compressed.values()) == [20, -1, 21, 22, 23, 32]
    routes = [(0, 5), (2, 5), (3, 7), (0, 6)]
    expert_m = Counter(e for pair in routes for e in pair)
    assert [expert_m[e] for e in range(8)] == [2, 0, 1, 1, 0, 2, 1, 1]
    assert sum(expert_m.values()) == 8
    remote = [(owner, expert // 2) for owner, pair in enumerate(routes)
              for expert in pair if expert // 2 != owner]
    remote_send = Counter(owner for owner, _ in remote)
    remote_recv = Counter(dest for _, dest in remote)
    assert len(remote) == 5
    assert [remote_send[r] * 8 for r in range(4)] == [8, 8, 16, 8]
    assert [remote_recv[r] * 8 for r in range(4)] == [8, 8, 16, 8]
    ar8 = ring_bytes("all_reduce", 1024 * 4096 * 2, 8)
    assert ar8["per_rank_send_MiB"] == 14
    assert ar8["group_send_B"] == 112 * 2**20
    ep_prefill = ring_bytes("all_gather", 24 * 4096 * 2, 8)
    ep_decode = ring_bytes("reduce_scatter", 16 * 4096 * 2, 8)
    assert ep_prefill["per_rank_send_B"] == 172032
    assert ep_decode["group_send_B"] == 917504
    assert (3 + 1) // 2 == 2  # SP source rows after padding to 4.
    assert 4 * 256 * 584 == 598016
    assert 2 * 512 * 528 == 540672
    page_content = 32 * 584
    page_aligned = (page_content + 575) // 576 * 576
    assert (page_content, page_aligned) == (18688, 19008)
    tp_groups = [[2 * dp, 2 * dp + 1] for dp in range(4)]
    dp_groups = [[2 * dp + tp for dp in range(4)] for tp in range(2)]
    report = {
        "evidence": "CPU arithmetic checks; no GPU or model computation",
        "config_origin": config_origin,
        "attention_shapes": shapes,
        "ordinary_slots": ordinary,
        "compressed_slots": compressed,
        "packed_page_teaching": {
            "records": 32, "content_B": page_content,
            "aligned_B": page_aligned, "alignment_B": 576,
        },
        "expert_m": [expert_m[e] for e in range(8)],
        "expert5_gemm_mnk": [[2, 12, 4], [2, 4, 6]],
        "tp_groups": tp_groups,
        "dp_groups": dp_groups,
        "ep_group": list(range(8)),
        "expert_parallel_sizes": {
            "EP_on": {"expert_tp": 1, "ep": 8},
            "EP_off": {"expert_tp": 8, "ep": 1},
        },
        "bf16_all_reduce_tp8": ar8,
        "bf16_ep_prefill_all_gather": ep_prefill,
        "bf16_ep_decode_reduce_scatter": ep_decode,
        "teaching_trace": {"serial": 15, "overlapped": 11},
        "source_sharing_teaching_content_bytes": {
            "v4": 598016, "v41": 540672, "includes_other_cache": False,
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded)
    print("PASS shapes, slots, routes, rank groups, ring bytes and trace arithmetic")


if __name__ == "__main__":
    main()
