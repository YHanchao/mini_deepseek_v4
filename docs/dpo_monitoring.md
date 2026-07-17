下面是可以直接交给 coding agent 的 markdown instruction。

# DPO Training Monitoring Metrics Enhancement

## Background

Current DPO implementation is working, but current monitoring metrics are insufficient to diagnose whether DPO is actually learning preference alignment.

Current implementation:

- Group size = 4 responses
- Construct all C(4,2)=6 preference pairs
- Preference is determined by synthetic reward scores
- DPO objective:

\[
L=-\log\sigma(\beta(\Delta))
\]

where:

\[
\Delta =
(\log\pi_\theta(y_w)-\log\pi_{ref}(y_w))
-
(\log\pi_\theta(y_l)-\log\pi_{ref}(y_l))
\]

Current code already reports:

- loss
- accuracy
- log_ratio_mean
- log_ratio_std
- log_ratio_diff_mean

However, `log_ratio_mean` is insufficient because it mixes chosen and rejected responses together.

---

# Required Changes

## 1. Add chosen/rejected log-ratio decomposition

Current:

```python
log_ratio = logp_policy - logp_ref
````

where:

[
log_ratio(y)
============

\log\pi_\theta(y)-\log\pi_{ref}(y)
]

For each preference pair, after selecting:

```python
chosen_lr
rejected_lr
```

add:

```python
chosen_log_ratio_mean
rejected_log_ratio_mean
```

Definition:

[
chosen_lr
=========

\log\pi_\theta(y_w)-\log\pi_{ref}(y_w)
]

[
rejected_lr
===========

\log\pi_\theta(y_l)-\log\pi_{ref}(y_l)
]

Aggregate over valid preference pairs.

Expected behavior during successful DPO:

```
chosen_log_ratio_mean:
    increase

rejected_log_ratio_mean:
    decrease
```

---

# 2. Add raw DPO margin

Currently:

```python
diff = beta * (chosen_lr - rejected_lr)
```

This mixes the training hyperparameter beta into the metric.

Add:

```python
raw_margin_mean
```

Definition:

[
margin =
chosen_lr-rejected_lr
]

and:

```python
scaled_margin_mean
```

Definition:

[
scaled_margin=
\beta\times margin
]

The existing:

```python
log_ratio_diff_mean
```

should be renamed to:

```python
scaled_margin_mean
```

or keep backward compatibility but additionally output raw margin.

Expected behavior:

```
raw_margin_mean:
    should increase during successful training
```

---

# 3. Add preference pair confidence statistics

For every valid pair:

```python
pair_prob = sigmoid(beta * raw_margin)
```

Report:

```python
pair_confidence_mean
```

Definition:

[
\sigma(\beta \Delta)
]

Interpretation:

* 0.5 = no preference information learned
* > 0.5 = model prefers chosen
* close to 1 = strong preference separation

---

# 4. Add train/validation split reporting

All metrics above should be available for:

* training batch
* validation set

Required logging:

```
train/dpo_loss
train/accuracy
train/raw_margin_mean
train/chosen_log_ratio_mean
train/rejected_log_ratio_mean

val/dpo_loss
val/accuracy
val/raw_margin_mean
val/chosen_log_ratio_mean
val/rejected_log_ratio_mean
```

---

# 5. Add KL-style drift metric (monitor only)

Although DPO does not explicitly optimize KL, monitor policy drift:

For each response:

[
KL_{approx}
===========

\log\pi_\theta-\log\pi_{ref}
]

Because this is sequence averaged, report:

```python
log_ratio_abs_mean
```

Definition:

[
mean(|log_ratio|)
]

Purpose:

Detect excessive deviation from reference.

Expected behavior:

* slowly increasing is normal
* sudden increase means instability

---

# 6. Keep existing metrics

Do not remove:

```
accuracy
log_ratio_mean
log_ratio_std
```

They are still useful.

---

# Final returned dictionary

The loss function should return at least:

```python
{
    "total_loss",

    "accuracy",

    # DPO signal
    "raw_margin_mean",
    "scaled_margin_mean",
    "pair_confidence_mean",

    # chosen/rejected separation
    "chosen_log_ratio_mean",
    "rejected_log_ratio_mean",

    # policy drift
    "log_ratio_mean",
    "log_ratio_std",
    "log_ratio_abs_mean",

    "logp_w",
    "logp_r",
}
```

---

# Logging recommendation

When printing validation results, use:

Example:

```
VAL:
loss=0.6928
acc=0.54

margin(raw)=0.012
margin(beta)=0.0012

chosen_lr=+0.006
rejected_lr=-0.006

confidence=0.501

log_ratio_mean=-0.001
log_ratio_abs=0.008
```

---

# Diagnostic interpretation

After implementation, evaluate DPO training using:

## Successful DPO

Expected:

```
accuracy ↑

raw_margin ↑

chosen_log_ratio ↑

rejected_log_ratio ↓

confidence ↑

log_ratio_abs slowly ↑
```

---

## No learning

```
accuracy ≈ 0.5

raw_margin ≈ 0

chosen_log_ratio ≈ rejected_log_ratio ≈ 0
```

---

## Wrong direction

```
accuracy ↓

raw_margin < 0

chosen_log_ratio < rejected_log_ratio
```

---

## Overtraining / divergence

```
log_ratio_abs rapidly ↑

margin rapidly ↑

validation accuracy ↓
```

---

# Implementation notes

Do not change the DPO objective.

Only add monitoring metrics.

Do not change:

* pair construction
* beta usage
* token log probability calculation
* sequence averaging method

This change is only for better diagnosis in future training runs.

```
```
