# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real scheduler with synthetic outputs and no model forward."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    str(Path(tempfile.gettempdir()) / "vllm-inference-walkthrough-cache"),
)

from transformers import OPTConfig  # noqa: E402

import torch  # noqa: E402

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import MLAAttentionSpec  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402


def make_scheduler(root, budget=6, blocks=64, prefix=False):
    model_dir = root / "model_config"
    model_dir.mkdir(parents=True, exist_ok=True)
    OPTConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        ffn_dim=64,
        num_attention_heads=4,
        max_position_embeddings=64,
        eos_token_id=2,
    ).save_pretrained(model_dir)
    return create_scheduler(
        model=str(model_dir),
        max_num_seqs=3,
        max_num_batched_tokens=budget,
        max_model_len=64,
        num_blocks=blocks,
        block_size=16,
        enable_prefix_caching=prefix,
        skip_tokenizer_init=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
    )


def add(scheduler, req_id, prompt, outputs):
    request = create_requests(
        num_requests=1,
        num_tokens=prompt,
        max_tokens=outputs,
        req_ids=[req_id],
        ignore_eos=True,
        same_prompt=True,
    )[0]
    scheduler.add_request(request)


def run_steps(scheduler, mode, budget, limit=100):
    trace = []
    for step in range(1, limit + 1):
        if mode == "mixed" and step == 2:
            add(scheduler, "P", 10, 2)
        if not scheduler.has_requests():
            break
        before = {
            r.request_id: {
                "computed": r.num_computed_tokens,
                "known": r.num_tokens,
                "preemptions": r.num_preemptions,
            }
            for r in scheduler.requests.values()
        }
        scheduled = scheduler.schedule()
        req_ids = list(scheduled.num_scheduled_tokens)
        computed_after = {
            rid: scheduler.requests[rid].num_computed_tokens
            for rid in req_ids
        }
        positions = {
            rid: list(range(computed_after[rid] - count, computed_after[rid]))
            for rid, count in scheduled.num_scheduled_tokens.items()
        }
        block_tables = {
            rid: scheduler.kv_cache_manager.get_block_ids(rid)
            for rid in req_ids
        }
        sampled = [
            [] if scheduler.requests[rid].is_prefill_chunk else [7]
            for rid in req_ids
        ]
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(scheduled, output)
        row = {
            "step": step,
            "before": before,
            "scheduled": dict(scheduled.num_scheduled_tokens),
            "positions": positions,
            "computed_after_schedule": computed_after,
            "block_tables": block_tables,
            "sampled_counts": dict(zip(req_ids, map(len, sampled))),
            "preempted": sorted(scheduled.preempted_req_ids),
            "remaining": {
                rid: {
                    "computed": req.num_computed_tokens,
                    "known": req.num_tokens,
                    "preemptions": req.num_preemptions,
                }
                for rid, req in scheduler.requests.items()
            },
            "free_blocks": (
                scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            ),
        }
        assert sum(row["scheduled"].values()) <= budget
        for rid, count in row["scheduled"].items():
            assert len(positions[rid]) == count
            assert row["sampled_counts"][rid] in (0, 1)
        trace.append(row)
    assert not scheduler.has_requests(), "The finite synthetic workload stalled."
    return trace


def exercise(root, mode, budget):
    if mode == "mixed":
        scheduler = make_scheduler(root, budget=budget)
        add(scheduler, "A", 2, 4)
        add(scheduler, "B", 2, 5)
        rows = run_steps(scheduler, mode, budget)
        if budget == 6:
            assert rows[1]["scheduled"] == {"A": 1, "B": 1, "P": 4}
            assert rows[2]["scheduled"] == {"A": 1, "B": 1, "P": 4}
            assert rows[3]["scheduled"] == {"A": 1, "B": 1, "P": 2}
            assert rows[1]["positions"]["P"] == [0, 1, 2, 3]
        return rows
    if mode == "pressure":
        scheduler = make_scheduler(root, budget=16, blocks=4)
        add(scheduler, "A", 16, 8)
        add(scheduler, "B", 16, 3)
        rows = run_steps(scheduler, mode, 16)
        assert any(row["preempted"] for row in rows), "No preemption observed."
        return rows
    scheduler = make_scheduler(root, budget=64, prefix=True)
    add(scheduler, "first", 32, 1)
    first = run_steps(scheduler, "prefix", 64)
    first_blocks = first[0]["block_tables"]["first"][0]
    add(scheduler, "second", 32, 1)
    second = run_steps(scheduler, "prefix", 64)
    hit = second[0]["positions"]["second"][0]
    assert hit == 16, "The full-block baseline must leave work for logits."
    assert second[0]["block_tables"]["second"][0][0] == first_blocks[0]
    return {"first": first, "second": second, "cached_tokens": hit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "mixed", "pressure", "prefix"],
                        default="all")
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()
    modes = ["mixed", "pressure", "prefix"] if args.mode == "all" else [args.mode]
    args.out.mkdir(parents=True, exist_ok=True)
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        tokens_per_state=4, state_content_bytes=584, alignment=576,
        model_version="deepseek_v4",
    )
    assert spec.num_states == 32
    assert spec.real_page_size_bytes == 18688
    assert spec.page_size_bytes == 19008
    (args.out / "cache_spec.json").write_text(
        json.dumps({
            "evidence": "real CPU KVCacheSpec; no GPU allocation",
            "block_size_original_tokens": 128,
            "tokens_per_state": 4,
            "records_per_page": spec.num_states,
            "content_B": spec.real_page_size_bytes,
            "aligned_page_B": spec.page_size_bytes,
        }, indent=2) + "\n"
    )
    for mode in modes:
        rows = exercise(args.out / mode, mode, args.budget)
        target = args.out / f"scheduler_{mode}.json"
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(f"PASS {mode}: real Scheduler, synthetic outputs -> {target}")


if __name__ == "__main__":
    main()
