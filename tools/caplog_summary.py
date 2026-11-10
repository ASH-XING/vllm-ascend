#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize LMHEAD_TP_CAP capacity-trace logs (instrument branch only).

The instrument branch emits one compact ``LMHEAD_TP_CAP|head|...`` line per
LM-head call (draft rounds and target), plus rare ``target_reqs_change``
batch-boundary lines and one-shot ``phase_mark`` lines. This tool collapses
them into:

- a per-step timeline for one rank (--rank);
- a cross-rank group-uniformity audit (default): every rank's N-th head call
  belongs to the same all-rank collective, so the per-rank value sequences
  must be identical element-wise — the "did every rank pad to the same
  capacity" check, independent of the per-rank step counters;
- a per-line self-consistency audit (default): pad == cap - rows, rows <= cap,
  and draft cap must equal the published dyn (or static after a drop);
- aggregate histograms: cap/rows/pad per side, formula (src) distribution,
  and the audit guards (static fallback / unpublished draft heads).

Usage (in the NPU container, on the serving log):

    python tools/caplog_summary.py serve.log --rank 2 --steps 4190:4200
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
    if value is None or value in ("None", "", "-"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def side_of(fields: dict[str, str]) -> str:
    side = fields.get("side", "")
    if side in ("t", "target"):
        return "target"
    if side in ("d", "draft") or side.startswith("draft"):
        return "draft"
    return "untagged"


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
        heads = []
        for tag, fields in by_step[step]:
            if tag in ("head", "head_consume"):
                heads.append(
                    f"[{side_of(fields)} rows={fields.get('rows')} "
                    f"dyn={fields.get('dyn')} static={fields.get('static')} "
                    f"cap={fields.get('cap')} pad={fields.get('pad')} "
                    f"src={fields.get('src')}]"
                )
            elif tag == "target_reqs_change":
                parts.append(f"REQS {fields.get('prev')}->{fields.get('num_reqs')}")
            elif tag == "phase_mark":
                parts.append(f"phase={fields.get('phase')}")
            else:
                parts.append(tag)
        print("  " + " | ".join(parts) + " " + " ".join(heads))


def histogram(counter: Counter, title: str, top: int = 12) -> None:
    print(f"  {title} ({sum(counter.values())} records):")
    for key, count in counter.most_common(top):
        print(f"    {key}: {count}")


def summary(records: list[tuple[str, str, dict[str, str]]]) -> None:
    tags = Counter(tag for _, tag, _ in records)
    ranks = Counter(rank for rank, _, _ in records)
    print("== record counts ==")
    print(f"  total LMHEAD_TP_CAP lines: {len(records)}")
    print(f"  ranks: {dict(sorted(ranks.items(), key=lambda kv: kv[0]))}")
    print(f"  tags: {dict(tags)}")

    phases = [
        f"rank {rank}: {fields.get('phase')}" for rank, tag, fields in records if tag == "phase_mark"
    ]
    if phases:
        print("== phase marks ==")
        for phase in phases:
            print(f"  {phase}")

    cap_hist = defaultdict(Counter)
    rows_hist = defaultdict(Counter)
    pad_hist = defaultdict(Counter)
    src_hist = defaultdict(Counter)
    reqs_changes = []
    for rank, tag, fields in records:
        if tag not in ("head", "head_consume"):
            if tag == "target_reqs_change":
                reqs_changes.append(
                    f"rank {rank} step {fields.get('step')}: {fields.get('prev')} -> {fields.get('num_reqs')}"
                )
            continue
        side = side_of(fields)
        cap_hist[side][fields.get("cap", "?")] += 1
        rows_hist[side][fields.get("rows", "?")] += 1
        pad_hist[side][fields.get("pad", "?")] += 1
        src_hist[side][fields.get("src", "?").split("=")[0]] += 1

    print("== head calls by side ==")
    for side in sorted(cap_hist):
        histogram(cap_hist[side], f"{side}: cap")
        histogram(rows_hist[side], f"{side}: rows", top=8)
        histogram(pad_hist[side], f"{side}: pad", top=8)
        print(f"    src formulas: {dict(src_hist[side])}")

    print("== request-wave boundaries (target_reqs_change) ==")
    shown = reqs_changes[:40]
    for change in shown:
        print(f"  {change}")
    if len(reqs_changes) > len(shown):
        print(f"  ... {len(reqs_changes) - len(shown)} more")


def cross_rank_check(records: list[tuple[str, str, dict[str, str]]]) -> None:
    """Group-uniformity audit: at every collective, did all ranks agree?

    Step counters drift between DP ranks, so steps cannot align ranks. The
    collective call sequence itself can: each rank's N-th head call belongs
    to the same all-rank collective as every other rank's N-th call of that
    stream. Every rank emits one head line per head call, so the per-rank
    sequences must be identical element-wise.
    """
    streams: dict[str, dict[str, list[object]]] = {}

    def add(stream: str, rank: str, value: object) -> None:
        streams.setdefault(stream, {}).setdefault(rank, []).append(value)

    for rank, tag, fields in records:
        if tag not in ("head", "head_consume"):
            continue
        side = side_of(fields)
        if side == "target":
            add("target head rows", rank, int_or_none(fields.get("rows")))
        else:
            add(
                "draft head cap,rows",
                rank,
                (int_or_none(fields.get("cap")), int_or_none(fields.get("rows"))),
            )

    print("== cross-rank group-uniformity (per collective sequence) ==")
    if not streams:
        print("  no head records found")
        return
    for name, per_rank in streams.items():
        ranks = sorted(per_rank)
        lens = {r: len(per_rank[r]) for r in ranks}
        n = min(lens.values()) if lens else 0
        mismatches = [i for i in range(n) if len({per_rank[r][i] for r in ranks}) > 1]
        counts_equal = len(set(lens.values())) == 1 if lens else True
        verdict = "ALL RANKS MATCH"
        if mismatches:
            verdict = f"{len(mismatches)} MISMATCHED COLLECTIVES"
        elif not counts_equal:
            verdict = "RECORD-COUNT DRIFT BETWEEN RANKS"
        print(f"  {name}: records/rank={sorted(lens.values())} -> {verdict}")
        if mismatches:
            for i in mismatches[:10]:
                grid = " ".join(f"DP{r}={per_rank[r][i]}" for r in ranks)
                print(f"    collective #{i + 1}: {grid}")
        elif not counts_equal:
            print(f"    per-rank counts: {lens} (a rank missed/added a call - investigate)")
        if n:
            sample = " ".join(f"DP{r}={per_rank[r][n - 1]}" for r in ranks)
            print(f"    last collective sample: {sample}")


def chain_check(records: list[tuple[str, str, dict[str, str]]]) -> None:
    """Per-line self-consistency: pad arithmetic and cap-vs-published checks.

    - pad must equal cap - rows, and rows must never exceed cap;
    - draft: cap must equal dyn when a dynamic value is published, else the
      static bound;
    - pub must equal step: healthy decode republishes every step, so a
      consume whose capacity was published at an earlier step means a
      keep-branch consume, a load-time static stamp, or a skipped publish;
    - target rows != cap is reported as info only: prefill / prompt-logprob
      consumes feed the real row count, only decode consumes are
      index-padded to cap.
    """
    bad_pad: list[tuple[str, str, object, object]] = []
    bad_rows: list[tuple[str, str, object, object]] = []
    bad_draft: list[tuple[str, str, object, object]] = []
    info_target: list[tuple[str, str, object]] = []
    stale: list[tuple[str, str, object, object]] = []

    for rank, tag, fields in records:
        if tag not in ("head", "head_consume"):
            continue
        side = side_of(fields)
        rows = int_or_none(fields.get("rows"))
        cap = int_or_none(fields.get("cap"))
        pad = int_or_none(fields.get("pad"))
        dyn = int_or_none(fields.get("dyn"))
        static = int_or_none(fields.get("static"))
        pub = int_or_none(fields.get("pub"))
        step = int_or_none(fields.get("step"))
        if pub is not None and step is not None and pub != step:
            # The consumed capacity was published at a different step: a
            # keep-branch consume (no fresh DP sync), a load-time static
            # stamp, or a skipped publish. Healthy decode always has
            # pub == step.
            stale.append((rank, f"src={fields.get('src', '?')}", pub, step))
        if cap is not None and pad is not None and pad != cap - rows:
            bad_pad.append((rank, "pad!=cap-rows", pad, None if rows is None else cap - rows))
        if cap is not None and rows is not None and rows > cap:
            bad_rows.append((rank, "rows>cap", rows, cap))
        if side == "draft":
            if dyn is not None:
                if cap != dyn:
                    bad_draft.append((rank, "cap!=dyn", cap, dyn))
            elif static is not None:
                if cap != static:
                    bad_draft.append((rank, "cap!=static", cap, static))
            else:
                bad_draft.append((rank, "unpublished draft head", cap, None))
        elif side == "target" and cap is not None and rows is not None and rows != cap:
            # Not necessarily a bug: prefill / prompt-logprob consumes feed the
            # real row count, only decode consumes are index-padded to cap.
            info_target.append((rank, "rows!=cap (prefill-like consume?)", fields.get("src", "?")))

    print("== per-line self-consistency ==")
    for name, issues in (
        ("pad != cap - rows", bad_pad),
        ("rows > cap", bad_rows),
        ("draft cap != published value", bad_draft),
    ):
        if issues:
            print(f"  {name}: {len(issues)} VIOLATIONS")
            for rank, what, got, want in issues[:10]:
                print(f"    rank {rank} {what}: got={got} expected={want}")
        else:
            print(f"  {name}: 0 (clean)")
    if info_target:
        print(f"  target rows != cap: {len(info_target)} (info, decode expects 0; prefill-like consumes)")
        for rank, what, src in info_target[:5]:
            print(f"    rank {rank} {what} src={src}")
    else:
        print("  target rows != cap: 0 (clean)")
    if stale:
        print(f"  pub != step (stale/kept capacity): {len(stale)} (healthy decode expects 0)")
        for rank, what, pub, step in stale[:10]:
            print(f"    rank {rank} published at step {pub}, consumed at step {step} {what}")
    else:
        print("  pub != step (stale/kept capacity): 0 (clean)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="serving log file")
    parser.add_argument("--rank", help="show per-step timeline for this DP rank (e.g. 2)")
    parser.add_argument("--steps", help="step range filter, e.g. 4190:4200 (timeline only)")
    args = parser.parse_args()

    records = load(args.log)
    if not records:
        sys.exit("no LMHEAD_TP_CAP records found in " + args.log)

    summary(records)
    print()
    cross_rank_check(records)
    print()
    chain_check(records)
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
