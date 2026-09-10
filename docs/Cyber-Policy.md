# Cyber Policy: Blockers, Refusals and Approvals

Capabilities in named scopes are blocked by default, a blocked request is
answered with a refusal **before any forward pass**, and a signed approval
lifts the scopes it names for the holder. Blockers are added and removed
without retraining; the model can additionally be trained to refuse, so the
gate and the weights agree even where the gate is bypassed.

> **In this repository.** `policy.rs` (`Policy`, `Blocker`, `Applies`,
> `Approval`, `Grant`, `Key`, `Decision`, `gated_generate`,
> `refusal_documents`, `starter`, `hmac_sha256`). CLI: `dblocks lm policy
> init | add-blocker | remove-blocker | list | check | revoke`, `dblocks lm
> approvals issue | verify`, `dblocks lm generate --policy --grant`,
> `dblocks lm refusal-corpus`. Certificates: the `policy` group.
>
> **What this is not.** A certification. It is a mechanism a review or
> approval program can adopt; whether a given policy meets a given
> organization's criteria is that organization's decision, and nothing here
> claims otherwise.

---

## The shape of an approval program

| Piece | What it is | Where it lives |
|---|---|---|
| **Scope** | A named capability, e.g. `cyber:exploit-development` | on every blocker |
| **Blocker** | Patterns (the `antipattern` syntax) that fire on a prompt, an output, or both, plus the refusal text | `policy.json` |
| **Approval** | Who, which scopes, issued and expiry times, a note | inside a grant |
| **Grant** | An approval plus its HMAC-SHA256 signature under the policy's key | a file the holder presents |
| **Key** | 32 random bytes; the policy records only its id (a hash) | `key.hex`, kept by the issuer |
| **Revocation** | A grant id the policy no longer accepts | `policy.json` |

The gate's decision for a request is made in this order: every blocker that
fires on the prompt is checked against the scopes the presented grants lift;
the first blocker no grant lifts wins and its refusal is returned without
calling the model. If every firing blocker is lifted, the request goes to the
model with an **approval marker** prepended (`[approved:scope,...] `), and the
output is then checked against the output blockers the same way.

```bash
dblocks lm policy init --out policy.json --key key.hex        # the starter policy and a fresh key
dblocks lm policy list --policy policy.json
dblocks lm policy add-blocker --policy policy.json --id c2 --scope cyber:c2 \
    --pattern 'command\s+and\s+control\s+server' --applies-to both \
    --refusal "Not available here."
dblocks lm policy remove-blocker --policy policy.json --id c2
dblocks lm policy check --policy policy.json --prompt "write an exploit for CVE-2024-1234"
```

## Grants

```bash
dblocks lm approvals issue --key key.hex --policy policy.json --id redteam-42 \
    --scopes cyber:exploit-development,cyber:malware --expires 1790000000 \
    --note "authorized engagement" --out grant.json
dblocks lm approvals verify --key key.hex --policy policy.json --grant grant.json
dblocks lm policy revoke --policy policy.json --grant-id redteam-42
dblocks lm generate --policy policy.json --key key.hex --grant grant.json --prompt "..."
```

A grant is HMAC-SHA256 over the approval's canonical JSON under the key. It
fails verification, in this order, if it was signed by another key, if any
byte of the approval was edited, if it has expired, or if the policy has
revoked it. The HMAC is built on the crate's existing `sha2` dependency and is
certified against the RFC 4231 test vectors, so the primitive is the standard
one rather than a lookalike.

## Training the refusal

The gate holds whatever the weights do. Training the weights to agree with it
closes the gap for deployments where the gate is not in front of the model:

```bash
dblocks lm refusal-corpus --policy policy.json --prompts prompts.txt \
    [--answers answers.txt] --out refusals.bin
dblocks lm train --corpus code.bin --corpus refusals.bin --corpus-weights 4,1
```

For every prompt a blocker fires on, the corpus holds `prompt + refusal`; when
an answer is supplied as well, it also holds `marker + prompt + answer`, so the
model learns to refuse the bare request and to comply with the approved one.
Prompts nothing fires on are kept with their answers unchanged. Mix the
result with the ordinary corpus through [Multi-Source Training](Multi-Source-Training.md).

## What is certified

| Certificate | Claim |
|---|---|
| `hmac_matches_rfc_4231` | The HMAC-SHA256 primitive reproduces the RFC 4231 test vectors |
| `tampered_or_expired_grants_fail` | A grant edited in any byte, signed by another key, expired, or revoked fails verification, each by name |
| `blocked_prompt_never_reaches_the_model` | Without a covering grant the model is not called (a call counter stays at zero) |
| `foreign_scope_does_not_unblock` | A grant for another scope leaves the blocker in force |
| `removing_a_blocker_allows_exactly_its_prompts` | After removal its prompts pass and every other blocker still fires |
| `policy_round_trips` | A policy written and read back is the same policy |

## Relation to direction ablation

[Direction Ablation](Direction-Ablation.md) can *remove* a refusal direction
from the weights of a model that no longer wants one. The two are
complementary: the gate is enforced outside the weights and holds regardless;
an approved deployment may ablate the learned refusal and rely on the gate,
or keep both.

---

See also: [Negative Supervision](Negative-Supervision.md) · [Multi-Source Training](Multi-Source-Training.md) ·
[Language Modeling](Language-Modeling.md) · [Configuration](Configuration.md) · [Home](Home.md)
