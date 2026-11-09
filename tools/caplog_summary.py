#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize LMHEAD_TP_CAP capacity-trace logs (instrument branch only).

The instrument branch emits ``LMHEAD_TP_CAP|<tag>|k=v|...`` records from every
DP-rank worker process. This tool collapses the interleaved wall of lines into:

- a per-(rank, step) timeline for one rank (--rank), showing the full capacity
  chain of each engine step: target publish -> target consume -> draft stash ->
  draft rounds (publish + head consume);
- aggregate summaries: capacity histograms per side, pad histograms, batch
  boundaries (target_reqs_change), phase marks, and the audit-guard counters
  (target_static_fallback / draft_dyn_drop / source=static hits).

Usage (in the NPU container, on the serving log):

    python tools/caplog_summary.py serve.log --rank 3 --steps 22660:22690
    python tools/caplog_summary.py serve.log            # summary only
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict

LINE_RE = re.compile(
    r"\((Worker_DP(\d+)_EP\d+|APIServer)[^)]*\)[^\n]*LMHEAD_TP_CAP\|([^|]+)\|(.*)$"
)


def parse_fields(payload: str) -> dict[str, str]:
    fields = {}
    for item in payload.split("|"):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        fields[key] = value
    return fields


def load(path: str) -> list[tuple[str, str, dict[str, str]]]:
    records = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE_RE.search(line)
            if not match:
                continue
            who, rank, tag, payload = match.groups()
            records.append((rank if who.startswith("Worker") else who, tag, parse_fields(payload)))
    return records


def int_or_none(value: str | None) -> int | None:
    if value is None or value in ("None", ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def fmt_consume(fields: dict[str, str]) -> str:
    capacity = fields.get("capacity", "?")
    rows = fields.get("rows", fields.get("kept", "?"))
    pad = fields.get("pad", "?")
    source = fields.get("source", "?")
    return f"cap={capacity} rows={rows} pad={pad} src={source}"


def step_timeline(
    records: list[tuple[str, str, dict[str, str]]], rank: str, lo: int | None, hi: int | None
) -> None:
    by_step: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    no_step: list[tuple[str, dict[str, str]]] = []
    for rec_rank, tag, fields in records:
        if rec_rank != rank:
            continue
        step = int_or_none(fields.get("step"))
        if step is None:
            no_step.append((tag, fields))
            continue
        if (lo is not None and step < lo) or (hi is not None and step > hi):
            continue
        by_step[fields["step"]].append((tag, fields))

    if no_step:
        print(f"== rank {rank}: records without step (load-time / phase) ==")
        for tag, fields in no_step:
            print(f"  {tag}: {' '.join(f'{k}={v}' for k, v in fields.items())}")
    print(f"== rank {rank}: per-step timeline ==")
    for step in sorted(by_step, key=int):
        parts = [f"step={step}"]
        draft_pub, draft_head = [], []
        for tag, fields in by_step[step]:
            if tag == "target_dyn_publish":
                parts.append(
                    f"tgt_pub dyn={fields.get('dyn')} "
                    f"(sync_reqs={fields.get('dp_sync_num_reqs')})"
                )
            elif tag == "target_static_fallback":
                parts.append(f"tgt_pub STATIC={fields.get('static')}")
            elif tag == "target_dyn_keep":
                parts.append(f"tgt_keep dyn={fields.get('kept')}")
            elif tag == "target_consume_sample":
                parts.append(f"tgt_sample {fmt_consume(fields)}")
            elif tag == "target_consume_dummy_join":
                parts.append(
                    f"tgt_dummy_join cap={fields.get('capacity')} src={fields.get('source')}"
                )
            elif tag == "target_reqs_change":
                parts.append(f"REQS {fields.get('prev')}->{fields.get('num_reqs')}")
            elif tag == "stash_num_reqs":
                parts.append(f"stash={fields.get('stashed')}")
            elif tag == "draft_dyn_publish":
                sync = fields.get("synced_max_reqs") or fields.get("synced_max_query_rows")
                draft_pub.append(f"{fields.get('dyn')}(sync={sync})")
            elif tag == "draft_dyn_drop":
                draft_pub.append("DROP")
            elif tag == "head_consume":
                side = fields.get("side", "?")
                entry = fmt_consume(fields)
                if side == "target":
                    parts.append(f"tgt_head {entry}")
                else:
                    draft_head.append(entry)
            else:
                parts.append(tag)
        if draft_pub:
            parts.append("draft_pub[" + ",".join(draft_pub) + "]")
        if draft_head:
            parts.append("draft_head[" + ",".join(draft_head) + "]")
        print("  " + " | ".join(parts))


def histogram(counter: Counter[int | str], title: str, top: int = 12) -> None:
    print(f"  {title} ({sum(counter.values())} records):")
    for key, count in counter.most_common(top):
        print(f"    {key}: {count}")


def summary(records: list[tuple[str, str, dict[str, str]]]) -> None:
    tags = Counter(tag for _, tag, _ in records)
    ranks = Counter(rank for rank, _, _ in records)
    print("== record counts ==")
    print(f"  total LMHEAD_TP_CAP lines: {len(records)}")
    print(f"  ranks: {dict(sorted(ranks.items(), key=lambda kv: kv[0]))}")
    print("  tags:")
    for tag, count in tags.most_common():
        print(f"    {tag}: {count}")

    target_dyn = Counter()
    target_source = Counter()
    draft_dyn = Counter()
    draft_round_sync = Counter()
    head_pad = defaultdict(Counter)
    head_rows = defaultdict(Counter)
    head_source = defaultdict(Counter)
    reqs_changes = []
    phases = []
    static_fallback_steps = defaultdict(list)
    drop_count = Counter()
    for rank, tag, fields in records:
        if tag == "target_dyn_publish":
            target_dyn[int_or_none(fields.get("dyn"))] += 1
            target_source["dynamic"] += 1
        elif tag == "target_static_fallback":
            target_source["static"] += 1
            static_fallback_steps[rank].append(fields.get("step"))
        elif tag == "target_dyn_keep":
            target_source["keep"] += 1
        elif tag == "target_reqs_change":
            reqs_changes.append(
                f"rank {rank} step {fields.get('step')}: {fields.get('prev')} -> {fields.get('num_reqs')}"
            )
        elif tag == "phase_mark":
            phases.append(f"rank {rank}: {fields.get('phase')}")
        elif tag == "draft_dyn_publish":
            dyn = int_or_none(fields.get("dyn"))
            if dyn is not None:
                draft_dyn[dyn] += 1
            sync = int_or_none(
                fields.get("synced_max_reqs") or fields.get("synced_max_query_rows")
            )
            draft_round_sync[sync] += 1
        elif tag == "draft_dyn_drop":
            drop_count[rank] += 1
        elif tag == "head_consume":
            side = fields.get("side", "?")
            head_pad[side][fields.get("pad", "?")] += 1
            head_rows[side][fields.get("rows", "?")] += 1
            head_source[side][fields.get("source", "?")] += 1

    print("== phase marks ==")
    for phase in phases:
        print(f"  {phase}")

    print("== target capacity ==")
    histogram(target_dyn, "dyn capacity")
    print(f"  source counts: {dict(target_source)}")

    print("== draft capacity ==")
    histogram(draft_dyn, "dyn capacity")
    histogram(draft_round_sync, "synced value (round 1 = tokens/req*k+1, later = reqs)", top=8)

    print("== head_consume by side ==")
    for side in sorted(head_rows):
        histogram(head_rows[side], f"{side}: rows")
        histogram(head_pad[side], f"{side}: pad", top=8)
        print(f"    source: {dict(head_source[side])}")

    print("== audit guards ==")
    if target_source.get("static", 0):
        head_steps = {r: v[:5] for r, v in static_fallback_steps.items()}
        print(f"  target_static_fallback: {target_source.get('static', 0)} (first steps per rank: {head_steps})")
    else:
        print("  target_static_fallback: 0 during logged window")
    print(f"  draft_dyn_drop: {sum(drop_count.values())} {dict(drop_count) if drop_count else ''}")
    draft_none = sum(head_source[side].get("none", 0) for side in head_source if side != "target")
    print(f"  draft head_consume source=none: {draft_none} (expected 0)")

    print("== request-wave boundaries (target_reqs_change) ==")
    shown = reqs_changes[:40]
    for change in shown:
        print(f"  {change}")
    if len(reqs_changes) > len(shown):
        print(f"  ... {len(reqs_changes) - len(shown)} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="serving log file")
    parser.add_argument("--rank", help="show per-step timeline for this DP rank (e.g. 3)")
    parser.add_argument("--steps", help="step range filter, e.g. 22660:22690 (timeline only)")
    args = parser.parse_args()

    records = load(args.log)
    if not records:
        sys.exit("no LMHEAD_TP_CAP records found in " + args.log)

    summary(records)
    if args.rank is not None:
        lo = hi = None
        if args.steps:
            lo_s, _, hi_s = args.steps.partition(":")
            lo = int(lo_s) if lo_s else None
            hi = int(hi_s) if hi_s else None
        print()
        step_timeline(records, args.rank, lo, hi)


if __name__ == "__main__":
    main()
