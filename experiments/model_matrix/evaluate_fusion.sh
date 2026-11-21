#!/usr/bin/env bash
set -euo pipefail

runtime=$1
bundle=$2
objective=$3
cases="$bundle/test.jsonl"

save_result() {
  local output=$1
  shift
  local temporary
  temporary=$(mktemp "$output.XXXXXX")
  if ! "$runtime" use --bundle "$bundle" "$@" > "$temporary"; then
    rm -- "$temporary"
    return 1
  fi
  mv -- "$temporary" "$output"
}

save_result "$bundle/heldout.json" --cases "$cases"
scene=$(head -n 1 "$cases" | jq -er .prompt)
reports=(heldout.json)
save_result "$bundle/demo.json" --scene "$scene"
cat "$bundle/demo.json"
if [[ "$objective" == diffusion ]]; then
  save_result "$bundle/heldout-denoised.json" --cases "$cases" --diffusion
  save_result "$bundle/demo-denoised.json" --scene "$scene" --diffusion
  cat "$bundle/demo-denoised.json"
  reports+=(heldout-denoised.json)
fi

for report in "${reports[@]}"; do
  result=$(jq -r '"\(.correct)/\(.cases) exact, kernel \(.kernel | tojson)"' "$bundle/$report")
  printf '%s %s %s\n' "$objective" "$report" "$result" |
    tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
done
