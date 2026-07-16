
# Add Ranking Metrics for Offline GRPO Validation

## Motivation

The current validation metrics (loss, KL, response log probability) are not sufficient for evaluating whether GRPO is actually learning the reward preference.

Offline GRPO optimizes preference ranking rather than language modeling loss.

Therefore, validation should directly measure whether the current policy ranks high-reward responses above low-reward responses.

---

# Inputs

Assume validation already computes

```python
logp_w      # (bs, group, seq_len)
scores      # (bs, group)
masks       # (bs, group, seq_len)
```

where

- `scores` are fixed reward labels.
- `group == 4`.

---

# Step 1: Compute response log probability

For each response:

```python
response_logp = (
    logp_w * masks.float()
).sum(dim=-1)
```

Shape:

```text
(bs, group)
```

---

# Step 2: Normalize by response length

Responses may have different lengths.

Use average token log probability instead of summed log probability.

```python
response_length = masks.float().sum(dim=-1).clamp(min=1)

response_score = response_logp / response_length
```

Shape:

```text
(bs, group)
```

This is the policy score used for ranking.

---

# Metric 1: Top-1 Accuracy

## Definition

For each group:

Find

```python
pred = response_score.argmax(dim=-1)
```

Find

```python
gt = scores.argmax(dim=-1)
```

Then

```python
top1_accuracy = (
    pred == gt
).float().mean()
```

---

## Log name

```text
val/rank_top1
```

---

## Interpretation

Random guessing (group size = 4):

```text
0.25
```

Perfect ranking:

```text
1.0
```

This is the easiest metric to interpret.

---

# Metric 2: Spearman Rank Correlation

## Definition

For every group:

Rank responses according to

- reward
- policy score

Compute Spearman rank correlation.

Average across the batch.

---

Recommended implementation:

```python
from scipy.stats import spearmanr
```

or implement manually if scipy is unavailable.

---

## Log name

```text
val/rank_spearman
```

---

## Interpretation

```text
1.0
```

Perfect ranking.

```text
0
```

Random.

```text
<0
```

Model prefers low-reward responses.

---

# Metric 3: Pairwise Ranking Accuracy

This metric uses every pair inside one group.

For each pair

```
(i, j)
```

If

```
reward_i > reward_j
```

then expect

```
policy_score_i > policy_score_j
```

Count correct pairs.

Average over all pairs.

For group size = 4

there are

```
6
```

pairs.

---

## Log name

```text
val/pairwise_accuracy
```

---

## Interpretation

Random:

```text
0.5
```

Perfect:

```text
1.0
```

This metric is usually more stable than Top-1.

---

# Metric 4: Reward-weighted Policy Score

Current implementation uses summed log probability.

Replace it with normalized score.

Current:

```python
reward_weighted = (
    response_logp * scores
).mean()
```

Replace with

```python
reward_weighted = (
    response_score * scores
).mean()
```

---

## Log name

```text
val/reward_weighted_score
```

---

# WandB Dashboard

Add four validation plots.

```text
val/rank_top1

val/rank_spearman

val/pairwise_accuracy

val/reward_weighted_score
```

---

# Expected Training Behaviour

Healthy training should show

- rank_top1 increasing
- pairwise_accuracy increasing
- rank_spearman increasing
- reward_weighted_score increasing

even if policy loss fluctuates.

If

- policy loss decreases
- ranking metrics remain unchanged

then GRPO is not actually learning the reward preference.

If

- KL increases rapidly
- ranking metrics decrease

then policy collapse is occurring.
