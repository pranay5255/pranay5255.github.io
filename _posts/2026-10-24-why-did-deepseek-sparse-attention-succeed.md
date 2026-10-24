---
layout: post
title: "Why did DeepSeek Sparse Attention succeed?"
date: "2026-10-24"
description: "A code-level walk through DeepSeek Sparse Attention: Lightning Indexer scores, global KV indices, sparse attention, and the compressed-KV trick."
categories: [writing]
tags: [deepseek, sparse-attention, gpu-kernels, mla]
reading_time: 6
math: true
---

DeepSeek Sparse Attention has gained a lot of traction as the go-to implementation for scaling attention and long context. Recently, the GLM team used DSA to compete with Claude Opus 4.5 successfully. So why is it successful?

DSA is a two-kernel pipeline for calculating attention. The first kernel is where the trick lies: the **Lightning Indexer**, which produces **top-k global KV indices**. We compute attention over a selected set of tokens rather than every token in the context.

## How to read this post

We'll understand DSA by mapping each term in the attention equation to the corresponding lines in the reference implementation. For each block: **term in the equation → exact code → what it means**.

## Lightning Indexer

### Compute index scores: ReLU → weighted head sum

The indexer ranks KV tokens and outputs **top-k global KV token indices**. These scores are pre-attention ranking signals. Index scores produce `topk_indices`: the candidate tokens to attend to for each query token.

<div class="math" markdown="0">
\[
I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j}\,\operatorname{ReLU}\!\left(q^I_{t,j}\cdot k^I_s\right)
\]
</div>

The left-hand side gives the raw scores used to rank tokens. This is a ranking function that decides which KV tokens deserve the full attention computation.

```python
scores = q_b @ K.T
scores_relu = torch.relu(scores)

w = weights[b]  # Per-sequence head weights in this implementation.
weighted_scores = scores_relu * w[:, None]
final_scores = weighted_scores.sum(dim=0)
```

### Why ReLU here?

For two simple reasons:

- ReLU keeps positive dot products and sets negative ones to zero before they contribute to a token's ranking score.
- ReLU is cheap to compute when the same operation must run repeatedly for every query.

### Top-k: convert local indices to global KV token indices

<div class="math" markdown="0">
\[
S = \operatorname{TopK}(I_{t,:}, K),\qquad
\operatorname{global}(i) = \operatorname{page\_id}(i)\cdot\operatorname{page\_size} + \operatorname{offset}(i)
\]
</div>

Top-k is computed in **local sequence-token space**. Sparse attention needs **global KV addresses**, so we convert using `global_page * page_size + offset`.

The KV cache is a flattened store with multiple pages. For the packed layout shown in the screenshots, you cannot simply split `[..., :128]` and `[..., 128:]` from the `[pages, 64, 1, 132]` view: that interleaves data and scales incorrectly. Flatten to `[pages, 64 * 132]` first, then split at the `64 * 128` boundary.

`topk_indices`, shaped `[B, 2048]` in this example, contains global flat indices into the KV cache. These become `sparse_indices` in stage two.

```python
actual_topk = min(topk, seq_len)
_, topk_idx = torch.topk(final_scores, actual_topk)

page_idx_per_token = topk_idx // page_size
offset_per_token = topk_idx % page_size
global_page_idx = page_indices[page_idx_per_token]
topk_tokens = global_page_idx * page_size + offset_per_token

topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
```

## Sparse attention

Stage two runs attention **only on the top-k indices selected** using the indexer's scores.

<div class="math" markdown="0">
\[
S_t = \{s_1,\ldots,s_K\},\qquad
K^c_t = K^c[S_t],\qquad K^{pe}_t = K^{pe}[S_t]
\]
</div>

- Kernel two flattens the KV cache into a global token array, and `sparse_indices` directly indexes it.
- This is where the first kernel's global KV indices pay off.

```python
Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

indices = sparse_indices[t]
valid_mask = indices != -1
valid_indices = indices[valid_mask]
tok_idx = valid_indices.to(torch.long)

Kc = Kc_all[tok_idx]
Kp = Kp_all[tok_idx]
```

### Compute logits: content dot product + positional dot product, then scale

<div class="math" markdown="0">
\[
\ell_{t,s} = \left\langle q^{nope}_t,K^c_s\right\rangle
+ \left\langle q^{pe}_t,K^{pe}_s\right\rangle,\qquad
\ell'_{t,s} = \ell_{t,s}\cdot\operatorname{sm\_scale}
\]
</div>

This is standard attention scoring, restricted to the selected set. In this example, stage two runs full attention over the 2,048 tokens selected by stage one. There are two variants: `page_size=1` for an unpaged cache and `page_size=64` for a paged cache. The core computation is identical.

```python
qn = q_nope[t].to(torch.float32)
qp = q_pe[t].to(torch.float32)

logits = (qn @ Kc.T) + (qp @ Kp.T)
logits_scaled = logits * sm_scale
```

### Core attention loop

```python
for t in range(num_tokens):
    indices = sparse_indices[t]              # [2048]
    valid_mask = indices != -1               # Filter padding.
    valid_indices = indices[valid_mask]

    # Gather: scattered reads from the KV cache.
    tok_idx = valid_indices.to(torch.long)
    Kc = Kc_all[tok_idx]                      # [num_valid, 512]
    Kp = Kp_all[tok_idx]                      # [num_valid, 64]

    qn = q_nope[t].to(torch.float32)           # [16, 512]
    qp = q_pe[t].to(torch.float32)             # [16, 64]

    # Split attention logit.
    logits = (qn @ Kc.T) + (qp @ Kp.T)        # [16, num_valid]
    logits_scaled = logits * sm_scale

    # Base-2 LSE, for merging with dense attention.
    lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

    # Softmax + output: the compressed KV tensor is also the value.
    attn = torch.softmax(logits_scaled, dim=-1)
    out = attn @ Kc                           # [16, 512]
    output[t] = out.to(torch.bfloat16)
```

## The final MLA trick: why V = K in compressed KV

DSA builds on **MLA: Multi-head Latent Attention**. In DeepSeek's MLA, both K and V are derived from the same compressed latent representation, `ckv`: the compressed KV cache.

<div class="math" markdown="0">
\[
\operatorname{score} = q_{nope}K_c^\top + q_{pe}K_{pe}^\top,\qquad
\operatorname{output} = \operatorname{softmax}(\operatorname{score})K_c
\]
</div>

There is no separate V tensor in this reference attention loop. The line `out = attn @ Kc` is intentional: compressed KV is used as the value. The 512-dimensional latent carries information for both key matching and value retrieval. Separate positional information goes through the `kpe` / `q_pe` path.

## Original article screenshots

<details>
<summary>View all nine original screenshots</summary>

<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/01-overview.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/01-overview.png' | relative_url }}" alt="Article introduction and the two-stage DSA pipeline: index selection, sparse indices, and attention."></a>
  <figcaption>1. Overview and the two-stage pipeline.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/02-lightning-indexer.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/02-lightning-indexer.png' | relative_url }}" alt="How to read the article, Lightning Indexer explanation, and the weighted ReLU score equation."></a>
  <figcaption>2. Lightning Indexer scores.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/03-indexer-code.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/03-indexer-code.png' | relative_url }}" alt="The indexer ranking equation and PyTorch code computing weighted per-head scores."></a>
  <figcaption>3. Indexer reference code.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/04-relu-and-global-indices.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/04-relu-and-global-indices.png' | relative_url }}" alt="Why the indexer uses ReLU and how local top-k indices map to global paged KV addresses."></a>
  <figcaption>4. ReLU and global KV indices.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/05-topk-and-sparse-attention.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/05-topk-and-sparse-attention.png' | relative_url }}" alt="Paged KV memory layout, top-k index conversion code, and the start of sparse attention."></a>
  <figcaption>5. Top-k selection and sparse attention.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/06-gather-and-logits.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/06-gather-and-logits.png' | relative_url }}" alt="Gathering selected compressed and positional keys, followed by the split attention-logit equation."></a>
  <figcaption>6. KV gathering and attention logits.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/07-scaled-attention.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/07-scaled-attention.png' | relative_url }}" alt="Content and positional dot products, attention scaling, and the start of the core attention loop."></a>
  <figcaption>7. Scaled attention scores.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/08-core-attention-loop.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/08-core-attention-loop.png' | relative_url }}" alt="Complete reference attention loop: gather KV entries, compute logits, log-sum-exp, softmax, and output."></a>
  <figcaption>8. Core attention loop.</figcaption>
</figure>
<figure>
  <a href="{{ '/assets/posts/deepseek-sparse-attention/09-mla-compressed-kv.png' | relative_url }}"><img loading="lazy" src="{{ '/assets/posts/deepseek-sparse-attention/09-mla-compressed-kv.png' | relative_url }}" alt="The MLA compressed-KV explanation showing the same compressed tensor used for scoring and values."></a>
  <figcaption>9. The compressed-KV trick.</figcaption>
</figure>
</details>
