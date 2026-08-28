# Language Modeling

A causal language model over the same trunk the image path uses. Byte-level
tokenization, a structurally tied head, block-wise spans, MoSME routing, a
key/value cache, and pre-tokenized corpora that stream from disk.

> **In this repository.** `tokenizer.rs` (`ByteTokenizer`, `Special`,
> `VOCAB_SIZE`), `corpus.rs` (`TokenCorpus`), `lm.rs` (`LmConfig`,
> `LanguageModel`, `KvCache`, `Sampling`, `LookaheadStats`). The causal mask
> lives in `vit.rs` (`causal_mask`, `causal_mask_offset`, `ViTDiTConfig::causal`).
> CLI: `dblocks lm tokenize | corpus | generate`. Certificates: the `lm` group.

---

## What is new and what is reused

The transformer layer is **not** duplicated. `LanguageModel` holds a `Vec<DbLayer>`
— the same layer the image trunk uses, with `causal: true`. That means MoE
routing, MoSME expert boxes, adaLN conditioning, span-wise execution and the
balance losses all apply to the language path with no second implementation to
keep in sync.

What is genuinely new:

| Piece | Why it could not be reused |
|---|---|
| Causal mask | Image patches have no ordering to respect; attention was unconditionally bidirectional |
| Token embedding | The only `Embedding` was indexed by a rank-1 *class label* tensor |
| Tied output head | The image head projects to `[batch, num_labels]` with no sequence axis |
| Next-token loss | Classification cross-entropy has one target per sample, not one per position |

---

## Tokenization

The vocabulary is the 256 byte values plus `<bos>`, `<eos>` and `<pad>`: 259
tokens, no vocabulary file, no merge table, no unknown tokens.

That is a real trade. Sequences are roughly 4× longer than with BPE on English
text, and attention is quadratic in length, so the same context costs more.
Against that: it is **lossless by construction** on every input including
arbitrary binary, there is no vocabulary artifact to version alongside a
checkpoint, and there are no normalization surprises.

The specials sit *above* the byte range rather than below it, so a raw byte's
token id is exactly its value:

```rust
let tok = ByteTokenizer::new();
assert_eq!(tok.encode("AZaz09"), vec![65, 90, 97, 122, 48, 57]);
```

A corpus file therefore stays readable with `xxd`, and a tokenization bug is
visible by eye.

`decode` returns `Option<String>` because a partially generated sequence often
is not valid UTF-8 — a multi-byte character can be cut in half. Streaming
callers want `decode_lossy`.

---

## Corpora

A corpus is a flat little-endian `u16` file with **no header**: token `i` is at
byte offset `2i`. Same fixed-stride shape as the image datasets in `rawdata.rs`,
for the same reason — any window is one seek, so a corpus larger than memory
costs no more per batch than one that fits.

```bash
dblocks lm tokenize --input book.txt --out book.bin
dblocks lm corpus --path book.bin --context 256
# book.bin: 1048578 tokens | 1048321 training windows at context 256
```

```rust
let mut corpus = TokenCorpus::streaming(Path::new("book.bin"))?;
let batch = corpus.sample_batch(32, 256, &mut rng)?;   // 32 windows of 257
```

Two decisions worth naming:

- **Windows are `context + 1` long.** Next-token training needs a target for the
  final input position. A corpus with fewer tokens than that yields *no* windows
  rather than a short one — training on a length the model will never see at
  inference is worse than training on nothing.
- **A short read is an error, not a pad.** Clamping would hand back a truncated
  sequence that downstream code pads and then counts as real tokens, quietly
  corrupting the loss denominator.

`in_memory` and `streaming` are certified to return identical windows, so the
choice is purely about memory. An odd byte count is rejected: it means a
truncated write or the wrong file, and reading it anyway would shift every token
after the damage.

---

## The model

```rust
let model = LanguageModel::<B>::new(&LmConfig::default(), &device);
let (loss, metrics) = model.next_token_loss(tokens, 0..model.num_layers());
println!("perplexity {:.2}", metrics.perplexity);

let grads = GradientsParams::from_grads(loss.backward(), &model);
let model = optimizer.step(lr, model, grads);
```

That whole loop — tokenize a file, stream windows out of the corpus, take AdamW
steps, watch the loss fall — is exercised by
`integration_a_language_model_trains_on_a_corpus` rather than asserted.

**The head is tied structurally**, not by copying weights: logits *are*
`h @ E^T` against the embedding table itself. There is no output projection that
could drift out of sync, and the vocabulary is paid for once.

That tying is also why the embedding initialization matters more than usual.
Burn's default `N(0, 1)` gave an untrained loss of **28.6** against the `ln(259)
= 5.56` a uniform model should show — the model started by having to undo its
own peaked logits. Initializing at `N(0, initializer_range)` fixes it, and
`lm/untrained_loss_is_near_uniform` keeps it fixed.

Padding is excluded from the loss **denominator**, not merely zeroed in the
numerator. A batch that is mostly padding would otherwise report a
reassuringly small loss that means nothing.

---

## Causality

The mask is additive `-inf` applied to the **scores**, before the softmax:

```rust
scores = scores + causal_mask::<B>(n, &device);
let probs = softmax(scores, 3);
```

Masking the probabilities afterwards would leave the denominator polluted by the
future — positions would still be normalized against weights they are not
allowed to use. Masking the scores makes `exp()` give exactly zero, so position
`i` can place no weight whatsoever on `i + 1`.

The certificate `lm/attention_cannot_see_the_future` perturbs token `i + 1` and
requires the logits at every position `≤ i` to be **bit-identical**, not merely
close.

---

## Generation and the KV cache

```bash
dblocks lm generate --prompt "Hello" --max-new 64 --cached
dblocks lm generate --prompt "Hello" --sampling topk --top-k 40 --temperature 0.8
```

Plain `generate` recomputes the whole prefix on every step: `O(n²)` per token.
`generate_cached` keeps each layer's keys and values and feeds only the new
token: `O(n)`.

**This is not a speed-for-accuracy trade.** A causal model's keys and values at
position `i` depend only on tokens up to `i`; once those are committed,
recomputing them can only reproduce the same numbers. So
`lm/kv_cache_matches_full_recompute` demands agreement to within float summation
order (`n·eps ≈ 1e-6` over the certificate's 16 positions), and
`lm/cached_decoding_emits_the_same_tokens` demands the emitted sequences be
identical.

One behavioural difference is deliberate. Past the context window, uncached
generation slides its window left and keeps going; a cache **stops**. Dropping
the oldest cached positions would invalidate every position embedding after
them, so the honest options are to stop or to rebuild the cache — and silently
rebuilding it would hide the cost.

```rust
let mut cache = model.new_cache();
let out = model.forward_cached(prompt_tokens, &mut cache);   // absorb the prompt
let next = model.forward_cached(one_token, &mut cache);      // one position each
cache.clear();                                               // reuse for another prompt
```

A cache is bound to one sequence. Reusing it without `clear` would silently
prepend the previous prompt — the kind of bug that surfaces as mysteriously
worse output rather than as an error, so `KvCache::position` is public and the
context assertion fires rather than truncating.

Lookahead decoding is on its own page: [Next-Step Planning](Next-Step-Planning.md).

---

## What is out of scope

Loading pretrained Llama- or Mistral-class weights. It needs safetensors or GGUF
parsing and a real tokenizer — dependencies this crate has deliberately avoided,
and a much larger surface than the LM path itself. The trunk here trains from
scratch.

---

See also: [Architecture](Architecture.md) · [Next-Step Planning](Next-Step-Planning.md) ·
[Mixture of Specialized Micro Experts](Mixture-of-Specialized-Micro-Experts.md) ·
[Quality Gate](Quality-Gate.md) · [Training Guide](Training-Guide.md)
