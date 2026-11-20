#!/bin/bash
# Full-send corpus scale-up: per-field ingest (open datasets) + quality_filter,
# ~50M tokens/field, resumable (skips a field whose -scale corpus already
# exists), then an all-fields mix for the shared tokenizer. Open datasets only.
# No `set -e`: one field failing must not abort the rest.
set -u
cd "/home/kali/Music/cybersec ai" || exit 1

# IMPORTANT: HF streaming buffers parquet shards to $TMPDIR. The system /tmp is
# a 12GB tmpfs (RAM-backed); big datasets (OpenMathInstruct-2, NuminaMath) filled
# it and tripped the low-memory reaper. Pin all temp/cache to the roomy root disk
# (53GB free) instead, so streaming never touches tmpfs/RAM.
export TMPDIR="/home/kali/Music/cybersec ai/nina-data/hf-tmp"
export HF_HOME="/home/kali/Music/cybersec ai/nina-data/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

PY=.venv-ingest/bin/python
MAP=unifur/cybercorpus/combined-open.map.json
RAW=nina-data/raw-scale
OUT=nina-data/field-corpora
LOG=nina-data/scale-progress.log
FIELDS="cyber coding math science tech"

: > "$LOG"
echo "=== scale run start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
for F in $FIELDS; do
  CORP="$OUT/${F}-scale.corpus.bin"
  if [ -s "$CORP" ]; then
    echo "$F  SKIP (corpus exists, $(( $(stat -c%s "$CORP")/2 )) tokens)" | tee -a "$LOG"
    continue
  fi
  echo ">>> $F ingest  $(date -u +%T)" | tee -a "$LOG"
  "$PY" -I unifur/cybercorpus/ingest_hf.py --field "$F" --map "$MAP" --out "$RAW" >> "$LOG" 2>&1
  echo ">>> $F filter  $(date -u +%T)" | tee -a "$LOG"
  "$PY" -I unifur/cybercorpus/quality_filter.py --paths "$RAW/$F" \
      --corpus-out "$CORP" --out "$RAW/${F}.clean.jsonl" >> "$LOG" 2>&1
  if [ -s "$CORP" ]; then
    echo "$F  DONE  $(( $(stat -c%s "$CORP")/2 )) tokens  $(date -u +%T)" | tee -a "$LOG"
  else
    echo "$F  FAILED (no corpus written)" | tee -a "$LOG"
  fi
done

# All-fields mix for the shared tokenizer: concatenate the per-field token
# streams (each is already BOS/bytes/EOS per doc, so a cat is a valid union).
MIX="$OUT/all-fields-scale.corpus.bin"
: > "$MIX"
for F in $FIELDS; do
  [ -s "$OUT/${F}-scale.corpus.bin" ] && cat "$OUT/${F}-scale.corpus.bin" >> "$MIX"
done
[ -s "$MIX" ] && echo "all-fields-mix  $(( $(stat -c%s "$MIX")/2 )) tokens" | tee -a "$LOG"
echo "=== scale run done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
