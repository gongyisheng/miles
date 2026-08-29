#!/bin/bash
# One-off preparation of the SFT dataset used by sft_common.sh.
#
# There is no ready-made `openhermes2_5.parquet` on the hub, so this downloads
# teknium/OpenHermes-2.5 (~1.9 GB of raw json) and converts its `conversations`
# (from/value) records into the `messages` (role/content) column that
# miles.rollout.sft_rollout expects.
#
#   bash examples/polora/sft_prepare_data.sh
#   MAX_ROWS=100000 bash examples/polora/sft_prepare_data.sh   # smaller output

set -ex

DATA_DIR=${DATA_DIR:-/root/datasets}
RAW_DIR=${RAW_DIR:-${DATA_DIR}/OpenHermes-2.5}
OUT_PATH=${OUT_PATH:-${DATA_DIR}/openhermes2_5.parquet}
MAX_ROWS=${MAX_ROWS:-0}   # 0 = keep every conversation

mkdir -p "${DATA_DIR}"

if [[ ! -f "${RAW_DIR}/openhermes2_5.json" ]]; then
   hf download --repo-type dataset teknium/OpenHermes-2.5 --local-dir "${RAW_DIR}"
fi

RAW_DIR="${RAW_DIR}" OUT_PATH="${OUT_PATH}" MAX_ROWS="${MAX_ROWS}" python3 - <<'PY'
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

raw_path = os.path.join(os.environ["RAW_DIR"], "openhermes2_5.json")
out_path = os.environ["OUT_PATH"]
max_rows = int(os.environ["MAX_ROWS"])

ROLES = {"system": "system", "human": "user", "gpt": "assistant"}
SCHEMA = pa.schema([("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])))])

print(f"loading {raw_path}")
with open(raw_path, encoding="utf-8") as f:
    records = json.load(f)
print(f"loaded {len(records)} conversations")

written = skipped = 0
with pq.ParquetWriter(out_path, SCHEMA) as writer:
    batch = []
    for record in records:
        turns = record.get("conversations") or []
        if any(turn.get("from") not in ROLES for turn in turns):
            skipped += 1
            continue
        messages = [{"role": ROLES[t["from"]], "content": t["value"]} for t in turns]
        # sft_rollout only produces a loss on assistant turns
        if not any(m["role"] == "assistant" for m in messages):
            skipped += 1
            continue
        batch.append(messages)
        written += 1
        if len(batch) == 10000:
            writer.write_table(pa.table({"messages": batch}, schema=SCHEMA))
            batch = []
        if max_rows and written >= max_rows:
            break
    if batch:
        writer.write_table(pa.table({"messages": batch}, schema=SCHEMA))

print(f"wrote {written} conversations to {out_path} (skipped {skipped})")
PY
