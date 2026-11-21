#!/usr/bin/env bash
set -euo pipefail

runtime=$1
bundle=$2
objective=$3
cases="$bundle/test.jsonl"

"$runtime" use --bundle "$bundle" --cases "$cases" > "$bundle/heldout.json"
scene=$(head -n 1 "$cases" | jq -er .prompt)
reports=(heldout.json)
"$runtime" use --bundle "$bundle" --scene "$scene" | tee "$bundle/demo.json"
if [[ "$objective" == diffusion ]]; then
  "$runtime" use --bundle "$bundle" --cases "$cases" --diffusion > "$bundle/heldout-denoised.json"
  "$runtime" use --bundle "$bundle" --scene "$scene" --diffusion | tee "$bundle/demo-denoised.json"
  reports+=(heldout-denoised.json)
fi

for report in "${reports[@]}"; do
  result=$(jq -r '"\(.correct)/\(.cases) exact, kernel \(.kernel | tojson)"' "$bundle/$report")
  printf '%s %s %s\n' "$objective" "$report" "$result" |
    tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
done
