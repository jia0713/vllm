# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify fixed source citations and local document links without importing vLLM."""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]


def main():
    data = json.loads((BASE / "source_refs.json").read_text())
    commit = data["commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO,
        check=True,
    )
    for ref in data["references"].values():
        raw = (REPO / ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"], ref["path"]
        lines = raw.decode().splitlines()
        assert 1 <= ref["start"] <= ref["end"] <= len(lines)
        assert lines[ref["start"] - 1].strip() == ref["declaration"], ref
        assert f"/blob/{commit}/" in ref["url"]
    critical_calls = {
        "corestep": [
            "self.scheduler.schedule(",
            "self.model_executor.execute_model(",
            "self.scheduler.update_from_output(",
        ],
        "schedschedule": ["self.kv_cache_manager.allocate_slots("],
        "schedadvance": ["request.num_computed_tokens += num_scheduled_token"],
        "runnerexecute": [
            "self._build_attention_metadata(",
            "set_forward_context(",
            "self._model_forward(",
        ],
        "v4prepare": [
            "self._wq_b_proj(", "self._fused_qnorm_rope_kv_insert(",
            "self._sparse_indexer_and_attn(",
        ],
        "v4sparse": ["self.indexer.indexer_op(", "self.forward_mqa("],
        "v4decoder": ["sp_all_gather(", "self.attn(", "sp_reduce_scatter("],
        "agdispatch": ["dist_group.all_gatherv("],
        "agcombine": ["dist_group.reduce_scatterv("],
        "v41cache": ["source.kv_cache"],
        "v41moe": ["num_hash_layers=0"],
    }
    for key, needles in critical_calls.items():
        ref = data["references"][key]
        lines = (REPO / ref["path"]).read_text().splitlines()
        region = "\n".join(lines[ref["start"] - 1:ref["end"]])
        for needle in needles:
            assert needle in region, (key, needle)
    links = 0
    diagrams = 0
    chapters = sorted(BASE.glob("[0-9][0-9]_*.md"))
    assert len(chapters) == 10
    for md in BASE.rglob("*.md"):
        text = md.read_text()
        assert not re.search(r"@@[a-z0-9_]+@@", text), md
        fences = re.findall(r"^~~~.*$", text, flags=re.MULTILINE)
        assert len(fences) % 2 == 0, md
        diagrams += text.count("~~~mermaid")
        for diagram in re.findall(r"~~~mermaid\n(.*?)\n~~~", text, re.S):
            rows = diagram.splitlines()
            assert rows[0] in ("flowchart LR", "flowchart TD"), md
            nodes = set(re.findall(r'\b([A-Za-z]\w*)\["', diagram))
            for row in rows[1:]:
                if not row.strip():
                    continue
                edge = re.sub(r'\["[^"]*"\]', "", row).strip()
                match = re.fullmatch(r"(\w+)\s*-->\s*(\w+)", edge)
                assert match, (md, row)
                assert set(match.groups()) <= nodes, (md, row)
        if md in chapters:
            assert "练习" in text and "answers.md#" in text, md
            assert text.count("~~~mermaid") >= 1, md
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("mailto:"):
                continue
            path_part, _, anchor = unquote(target).partition("#")
            path = (md.parent / path_part).resolve() if path_part else md
            assert path.exists(), (md, target)
            if anchor and re.fullmatch(r"[0-9]{2}", anchor):
                assert f'id="{anchor}"' in path.read_text(), (md, target)
            links += 1
    for script in (BASE / "scripts").glob("*.py"):
        source = script.read_text()
        ast.parse(source, filename=str(script))
        assert source.startswith("# SPDX-License-Identifier: Apache-2.0")
        assert source.endswith("\n"), script
        assert all(len(line) <= 88 for line in source.splitlines()), script
        assert all(line == line.rstrip() for line in source.splitlines()), script
    report = {
        "commit": commit,
        "source_references_checked": len(data["references"]),
        "critical_call_sites_checked": sum(
            map(len, critical_calls.values())
        ),
        "local_links_checked": links,
        "chapters": len(chapters),
        "mermaid_blocks": diagrams,
        "scripts_parsed": len(list((BASE / "scripts").glob("*.py"))),
        "evidence": "static citation/link/syntax verification; no GPU execution",
    }
    (BASE / "validation" / "document_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PASS source snapshots, declaration lines, local links and chapter structure")


if __name__ == "__main__":
    main()
