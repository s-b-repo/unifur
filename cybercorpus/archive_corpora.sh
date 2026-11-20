#!/bin/bash
# Copy every FINISHED scaled corpus into a permanent archive subfolder, with
# clean names, the run log, the dataset map, and a generated manifest. Idempotent
# and re-runnable: run it any time to snapshot whatever fields have completed so
# far (safe to run while the scale job is still building the remaining fields).
set -u
cd "/home/kali/Music/cybersec ai" || exit 1
SRC=nina-data/field-corpora
PERM=nina-data/corpora-scaled
LOG=nina-data/scale-progress.log
MAP=unifur/cybercorpus/combined-open.map.json
FIELDS="cyber coding math science tech"
mkdir -p "$PERM"

copied=0
for f in $FIELDS; do
  s="$SRC/${f}-scale.corpus.bin"
  [ -s "$s" ] && cp -f "$s" "$PERM/${f}.corpus.bin" && copied=$((copied+1))
  v="$SRC/${f}-scale.val.corpus.bin"
  [ -s "$v" ] && cp -f "$v" "$PERM/${f}.val.corpus.bin"
done
[ -s "$SRC/all-fields-scale.corpus.bin" ] && cp -f "$SRC/all-fields-scale.corpus.bin" "$PERM/all-fields.corpus.bin"
[ -s "$SRC/all-fields-scale.val.corpus.bin" ] && cp -f "$SRC/all-fields-scale.val.corpus.bin" "$PERM/all-fields.val.corpus.bin"
[ -s "$LOG" ] && cp -f "$LOG" "$PERM/scale-progress.log"
[ -s "$MAP" ] && cp -f "$MAP" "$PERM/datasets.map.json"

# Manifest: token counts per field + the filter rates pulled from the run log.
M="$PERM/MANIFEST.md"
{
  echo "# Scaled NCM per-field corpora"
  echo
  echo "Permanent archive. Headerless little-endian u16 token streams (BOS=256,"
  echo "EOS=257 per document), as \`dblocks lm train --corpus\` reads. Open HF"
  echo "datasets only (no auth). Snapshot: $(date -u +%FT%TZ)."
  echo
  echo "| Field | Tokens | Bytes |"
  echo "|---|---|---|"
  total=0
  for f in $FIELDS all-fields; do
    c="$PERM/${f}.corpus.bin"
    if [ -s "$c" ]; then
      b=$(stat -c%s "$c"); t=$((b/2))
      [ "$f" != "all-fields" ] && total=$((total+t))
      printf "| %s | %s | %s |\n" "$f" "$t" "$b"
    fi
  done
  echo
  echo "Per-field total (excl. all-fields mix): **$total tokens**."
  echo
  echo "## Filter rates (from the run log)"
  echo '```'
  grep -E 'seen [0-9]+|secrets:|smuggle:|DONE|SKIP' "$LOG" 2>/dev/null
  echo '```'
  echo
  echo "Datasets used: see datasets.map.json. Full run log: scale-progress.log."
} > "$M"

echo "archived $copied field corpora -> $PERM"
ls -la "$PERM" | awk 'NR>1{print $5, $9}'
