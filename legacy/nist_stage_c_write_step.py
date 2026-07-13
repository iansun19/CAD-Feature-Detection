#!/usr/bin/env python3
"""
Stage C (pure text, no OCC): write predicted class ids into the 139 ADVANCED_FACE
name fields of fixtures/step/fixtures/step/nist_ctc_01.step, producing fixtures/step/fixtures/step/nist_ctc_01_annotated.step. MFCAD++
format: bare integer in the name field, e.g. ADVANCED_FACE('11',(...). Every other
byte is preserved. Map comes from nist_ctc_01_predictions.jsonl (entity #N -> class).

Run: .venv/bin/python nist_stage_c_write_step.py
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "fixtures/step/fixtures/step/nist_ctc_01.step")
DST = os.path.join(ROOT, "fixtures/step/fixtures/step/nist_ctc_01_annotated.step")
JSONL = os.path.join(ROOT, "nist_ctc_01_predictions.jsonl")

# Match "#<N>=ADVANCED_FACE('<name>'," capturing pieces so we replace only the name.
# Bytes, not text: the source uses CRLF and must be preserved byte-for-byte.
AF_RE = re.compile(rb"(#(\d+)\s*=\s*ADVANCED_FACE\(')([^']*)(')")


def main():
    ent_to_cls = {}
    with open(JSONL) as f:
        for line in f:
            r = json.loads(line)
            ent_to_cls[int(r["entity_id"])] = int(r["class_id"])
    print(f"[Stage C] loaded {len(ent_to_cls)} entity->class predictions")

    with open(SRC, "rb") as f:
        data = f.read()

    written = {"n": 0}
    ambiguous = []

    def repl(m):
        eid = int(m.group(2))
        if eid not in ent_to_cls:
            ambiguous.append(eid)
            return m.group(0)  # leave untouched
        if m.group(3) != b"":
            ambiguous.append(eid)  # unexpected pre-existing name
        written["n"] += 1
        return m.group(1) + str(ent_to_cls[eid]).encode() + m.group(4)

    new_data = AF_RE.sub(repl, data)

    with open(DST, "wb") as f:
        f.write(new_data)
    print(f"[Stage C] wrote {DST}")
    print(f"[Stage C] annotated {written['n']} ADVANCED_FACE name fields")
    if ambiguous:
        print(f"[Stage C] !! AMBIGUOUS/unexpected entities: {sorted(set(ambiguous))}")
    else:
        print("[Stage C] no ambiguous mappings")

    # byte-diff sanity: only name-field regions changed, line count identical
    src_lines, dst_lines = data.split(b"\n"), new_data.split(b"\n")
    assert len(src_lines) == len(dst_lines), "line count changed!"
    changed = sum(1 for a, b in zip(src_lines, dst_lines) if a != b)
    cr_src, cr_dst = data.count(b"\r"), new_data.count(b"\r")
    print(f"[Stage C] changed lines: {changed} (expect == 139); "
          f"total lines {len(src_lines)} unchanged in count")
    print(f"[Stage C] CRLF preserved: source CR={cr_src} annotated CR={cr_dst}")


if __name__ == "__main__":
    main()
