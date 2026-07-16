# GRPO Training Monitoring Metrics Specification

## Purpose

Implement monitoring metrics for offline GRPO training.

The goal is to detect:

1. Policy collapse
2. Excessive deviation from reference model
3. PPO update instability
4. Reward overfitting
5. Loss decreasing while model quality deteriorating

All metrics should be logged to wandb.

---

# Available tensors

Assume the following tensors exist inside `grpo_loss` or training loop:

```python
logp_w      # (bs, gs, seq_len)
            # current policy token log probabilities

logp_r      # (bs, gs, seq_len)
            # reference model token log probabilities

log_ratio   # (bs, gs, seq_len)
            # logp_w - logp_r

ratio       # (bs, gs, seq_len)
            # exp(log_ratio)

scores      # (bs, gs)
            # reward scores

masks       # (bs, gs, seq_len)
            # 1 for completion tokens, 0 otherwise

adv         # (bs, gs, 1)
            # normalized GRPO advantage

valid_tokens = masks.float().sum()


# 1. Policy Loss

## Name
```

train/policy_loss

```

## Formula

```python
ratio_clipped = torch.clamp(
    ratio,
    1 - clip_eps,
    1 + clip_eps
)

gain = torch.min(
    ratio * adv,
    ratio_clipped * adv
)

policy_loss = -(
    gain * masks.float()
).sum() / valid_tokens
```

## Purpose

Monitor PPO objective.

Do NOT use this metric alone to judge training quality.

A decreasing policy loss does not necessarily mean better model performance.

---

# 2. KL Divergence

## Name

```
train/kl
```

## Formula

Use DeepSeek k3 estimator:

```python
kl_per_token = ratio - log_ratio - 1

kl = (
    kl_per_token * masks.float()
).sum() / valid_tokens
```

## Purpose

Measure deviation from reference model.

This is the most important stability metric.

Expected behavior:

```
start: approximately 0

training:
slow increase
```

Warning:

```
KL continuously increasing
```

means policy is drifting away from reference.

---

# 3. KL Penalty Contribution

## Names

```
train/kl_penalty
```

## Formula

```python
kl_penalty = beta * kl
```

## Purpose

Measure how much KL regularization contributes to total loss.

Compare:

```
policy_loss
vs
beta * kl
```

---

# 4. Mean Importance Ratio

## Name

```
train/ratio_mean
```

## Formula

```python
ratio_mean = (
    ratio * masks.float()
).sum() / valid_tokens
```

## Purpose

Measure average policy change.

Expected:

```
approximately 1.0
```

Warning:

```
ratio_mean >> 1
```

indicates policy drift.

---

# 5. Ratio Standard Deviation

## Name

```
train/ratio_std
```

## Formula

```python
ratio_valid = ratio[masks.bool()]

ratio_std = ratio_valid.std()
```

## Purpose

Detect unstable token-level updates.

Large increase indicates some tokens receive excessive updates.

---

# 6. Maximum Importance Ratio

## Name

```
train/ratio_max
```

## Formula

```python
ratio_max = ratio[masks.bool()].max()
```

## Purpose

Detect extreme token probability changes.

Warning:

```
ratio_max > 5
```

Severe:

```
ratio_max > 10
```

---

# 7. PPO Clip Fraction

## Name

```
train/clip_fraction
```

## Formula

```python
clip_mask = (
    (ratio > 1 + clip_eps)
    |
    (ratio < 1 - clip_eps)
)

clip_fraction = (
    clip_mask.float()
    *
    masks.float()
).sum() / valid_tokens
```

## Purpose

Measure how often PPO clipping activates.

Interpretation:

```
<0.1
healthy

0.1 - 0.5
large updates

>0.5
training too aggressive
```

---

# 8. Reward Statistics

## Names

```
train/reward_mean
train/reward_std
train/reward_max
train/reward_min
train/reward_margin
```

## Formula

```python
reward_mean = scores.mean()

reward_std = scores.std()

reward_max = scores.max()

reward_min = scores.min()

reward_margin = (
    scores.max(dim=-1).values
    -
    scores.min(dim=-1).values
).mean()
```

## Purpose

Monitor reward signal.

Important because GRPO depends on within-group reward differences.

---

# 9. Advantage Statistics

## Names

```
train/adv_mean
train/adv_std
train/adv_max
train/adv_min
```

## Formula

```python
adv_mean = adv.mean()

adv_std = adv.std()

adv_max = adv.max()

adv_min = adv.min()
```

Expected:

```
adv_mean ≈ 0

adv_std ≈ 1
```

---

# 10. Policy Entropy

## Names

```
train/entropy
val/entropy
```

## Formula

Given model logits:

```python
log_probs = torch.log_softmax(logits_working, dim=-1)

probs = torch.exp(log_probs)

entropy_per_token = -(
    probs * log_probs
).sum(dim=-1)

entropy = (
    entropy_per_token * masks.float()
).sum() / valid_tokens
```

## Purpose

Detect model becoming deterministic.

Warning:

Rapid entropy decrease means possible collapse.

Example:

```
entropy:
5.0
4.5
2.0
```

---

# 11. Validation Metrics

Do NOT use GRPO loss as the main validation metric.

For validation set, compute:

## 11.1 Validation KL

Name:

```
val/kl
```

Formula:

same as training KL.

---

## 11.2 Validation Reward-Weighted Log Probability

Name:

```
val/reward_weighted_logp
```

Formula:

```python
seq_logp = (
    logp_w * masks.float()
).sum(dim=-1)

metric = (
    seq_logp * scores
).mean()
```

Purpose:

Check whether model assigns higher probability to high reward responses.

---

## 11.3 Validation Response Log Probability

Name:

```
val/response_logp
```

Formula:

```python
seq_logp = (
    logp_w * masks.float()
).sum(dim=-1)

metric = seq_logp.mean()
```

---

# Required wandb dashboard

Create the following panels:

## Loss

```
train/policy_loss
train/kl
train/kl_penalty
train/total_loss
```

## PPO Stability

```
train/ratio_mean
train/ratio_std
train/ratio_max
train/clip_fraction
```

## Reward

```
train/reward_mean
train/reward_std
train/reward_margin
```

## Language Quality

```
train/entropy
val/entropy
```

## Validation

```
val/kl
val/reward_weighted_logp
val/response_logp
```

---

# Training failure diagnosis rules

## Case 1: KL rapidly increases

Symptoms:

```
train/kl ↑↑
ratio_max ↑↑
clip_fraction ↑↑
entropy ↓
```

Diagnosis:

Policy collapse.

Actions:

* decrease learning rate
* increase KL coefficient
* reduce training steps

---

## Case 2: Loss decreases but validation worsens

Symptoms:

```
train/policy_loss ↓
val/reward_weighted_logp ↓
KL ↑
```

Diagnosis:

Offline overfitting.

Actions:

* fewer epochs
* stronger KL
* smaller LR

---

## Case 3: Nothing changes

Symptoms:

```
KL ≈ 0
ratio_mean ≈ 1
clip_fraction ≈ 0
reward unchanged
```

Diagnosis:

Updates too weak.

Actions:

* increase LR
* decrease KL coefficient

```
```
