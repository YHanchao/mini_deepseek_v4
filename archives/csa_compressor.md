# CSA Compressor 重叠窗口与 kv_state 运作详解

## 1. 符号约定与极简场景

为了看清张量变换的细节，假设：

- 压缩比率 `compress_ratio = m = 4`
- 头维度 `head_dim = d = 3`（实际为 512，用 3 便于示例）
- `overlap = True` → `coff = 2`，每个 token 的投影向量为 `2d = 6` 维
- 序列长度 `n = 10`（预填充时一次输入 10 个 token）

**向量结构：**  
由 `wkv` 产生的 KV 向量（同样适用于 `wgate` 产生的 score）维度为 `[6]`：
```
Token t:  [ aₜ⁰ aₜ¹ aₜ² | bₜ⁰ bₜ¹ bₜ² ]
           ← 前半 :d=3 →   ← 后半 d:=3 →
```

**代码中的角色分工（关键）：**
- **前半 `:d`（代码中的 a 部分）**：用于跨块重叠，功能上等价于论文的 **Cᵇ / Zᵇ**。
- **后半 `d:`（代码中的 b 部分）**：当前块自身的表示，功能上等价于论文的 **Cᵃ / Zᵃ**。

因此，在重叠模式下，一个压缩窗口由 **前一块的 a 部分（Cᵇ）** 和 **当前块的 b 部分（Cᵃ）** 拼接而成。

---

## 2. `overlap_transform` 的错位拼接

**输入：** `tensor` 形状 `[b, s, r, 2d]`  
- `s`：压缩块数（`seqlen // r`）  
- `r = m = 4`，`d = 3`

**输出：** `new_tensor` 形状 `[b, s, 2r, d]`

### 两步赋值

```python
new_tensor[:, :, ratio:] = tensor[:, :, :, d:]      # ①
new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]   # ②
```

#### ① 将“当前块的后半 (b/Cᵃ)”放入窗口后半
- 右端 `tensor[:, :, :, d:]`：所有块的后 `d` 维（b 部分），形状 `[b, s, r, d]`
- 左端 `new_tensor[:, :, ratio:]`：每个压缩窗口的后 `r` 个位置（索引 `r..2r-1`）

#### ② 将“前一块的前半 (a/Cᵇ)”放入窗口前半
- 右端 `tensor[:, :-1, :, :d]`：前 `s-1` 个块的前 `d` 维（a 部分），形状 `[b, s-1, r, d]`
- 左端 `new_tensor[:, 1:, :ratio]`：**从第 1 个窗口开始**的前 `r` 个位置（索引 `0..r-1`）

对于第 0 个窗口，前半保持初始填充值（KV 为 0，score 为 `-inf`），等价于论文 `i=0` 时 Cᵇ 填零、Zᵇ 填负无穷。

### 示例：2 个原始块 → 2 个压缩窗口
原始块 0（token 0,1,2,3），原始块 1（token 4,5,6,7）

- **窗口 0**：前 `r` 个为 0/-inf，后 `r` 个 = 块 0 的 b 部分（b₀..b₃）
- **窗口 1**：前 `r` 个 = 块 0 的 a 部分（a₀..a₃），后 `r` 个 = 块 1 的 b 部分（b₄..b₇）

这精确对应论文公式(11)的拼接：
```
[ Zᵇ_{m(i-1):mi-1} + Bᵇ ; Zᵃ_{mi:m(i+1)-1} + Bᵃ ]
```
softmax 在 8 个元素上归一化后，与对应 KV 加权求和，得到压缩后的 `C_i^Comp`。

---

## 3. 预填充阶段如何为解码准备 `kv_state`

预填充输入 10 个 token，`cutoff = 8`，`remainder = 2`。

### 3.1 存储最后一个完整块（用于重叠接力）
```python
if overlap and cutoff >= ratio:
    self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
    self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
```
取 token 4,5,6,7（最后一个完整块）的**完整 6 维向量**，存入 `kv_state` 的前 4 行（索引 0..3）。  
解码阶段重叠窗口的前半部分（Cᵇ）会从这些行的**前半 `:d`（a 部分）** 提取。

### 3.2 暂存余数 token
```python
if remainder > 0:
    kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split(...)
```
`offset = ratio = 4`，余数 token 8,9 被存入 `kv_state` 的第 4、5 行（后 4 行的开头）。  
它们将在解码时继续累积，凑满 4 个后参与下一次压缩。

### 预填充后 `kv_state` 内容（batch=0）

| 行索引 | 存储内容（6维）      | 角色               |
|--------|---------------------|--------------------|
| 0      | Token 4 (a₄,b₄)     | 前一块（完整）     |
| 1      | Token 5 (a₅,b₅)     |                    |
| 2      | Token 6 (a₆,b₆)     |                    |
| 3      | Token 7 (a₇,b₇)     |                    |
| 4      | Token 8 (a₈,b₈)     | 当前块累积（余数） |
| 5      | Token 9 (a₉,b₉)     |                    |
| 6      | -inf                | 尚未使用           |
| 7      | -inf                |                    |

> `kv_state` 总大小 = `coff * ratio = 8` 行。前 4 行固定用于“前一块”信息，后 4 行用于“当前块”累积。

---

## 4. 解码阶段 `kv_state` 的滚动更新

预填充结束后，`kv_cache` 中已有压缩后的 token 0..7。接下来逐 token 解码。

### 4.1 Token 10（start_pos=10）
- `start_pos % 4 = 2`，`should_compress = False`
- 存入 `kv_state[ratio + 2]` → 第 6 行

| 行 | 内容     |
|----|----------|
| 0 | token 4  |
| 1 | token 5  |
| 2 | token 6  |
| 3 | token 7  |
| 4 | token 8  |
| 5 | token 9  |
| 6 | **token 10** |
| 7 | -inf     |

不满足压缩条件，直接返回。

### 4.2 Token 11（start_pos=11）
- `start_pos % 4 = 3`，`should_compress = True`
- 存入 `kv_state[7]`

| 行 | 内容     |
|----|----------|
| 0 | token 4  |
| 1 | token 5  |
| 2 | token 6  |
| 3 | token 7  |
| 4 | token 8  |
| 5 | token 9  |
| 6 | token 10 |
| 7 | **token 11** |

触发压缩：构造 8 个向量的拼接窗口：
- 前 4 个：取 `kv_state[0:4]` 的**前半 `:d`（a 部分）** → token 4..7 的 a 部分 → 论文 Cᵇ
- 后 4 个：取 `kv_state[4:8]` 的**后半 `d:`（b 部分）** → token 8..11 的 b 部分 → 论文 Cᵃ

softmax 加权求和 → 一个压缩向量 → 写入 `kv_cache` 的对应位置。

### 4.3 滚动移位
```python
self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
```
将后 4 行（token 8,9,10,11）复制到前 4 行。`kv_state` 变为：

| 行 | 内容     |
|----|----------|
| 0 | token 8  |
| 1 | token 9  |
| 2 | token 10 |
| 3 | token 11 |
| 4..7 | （待覆盖） |

至此，刚压缩完的 4 个 token 成为下一轮的“前一块”。循环继续。

### 4.4 后续 token（12..15）
token 12 存入第 4 行，13 存入第 5 行，14 存入第 6 行，15 存入第 7 行。  
到 token 15 时再次凑满 4 个，触发压缩：
- 前 4 个：token 8..11 的 a 部分（Cᵇ）
- 后 4 个：token 12..15 的 b 部分（Cᵃ）
压缩后再次移位，如此反复。

---

## 5. 总结

- **`overlap_transform`**：通过张量索引的错位赋值，实现论文中“前一块的 Cᵇ + 当前块的 Cᵃ”的拼接，并正确处理边界填充。
- **`kv_state` 结构**：一个 8 行的缓存，前 4 行永久保存“前一块”的完整 token，后 4 行累积“当前块”的 token。
- **预填充的初始化**：截取最后一个完整块放入前 4 行，余数放入后 4 行，为流式解码奠定状态基础。
- **解码滚动**：每次凑齐 `ratio` 个 token 后，提取前 4 行的 a 部分与后 4 行的 b 部分拼接压缩，再将后 4 行整体复制到前 4 行，完成状态的更新。
- **与论文公式的对应**：代码中 a 部分 ↔ 论文 Cᵇ/Zᵇ，b 部分 ↔ 论文 Cᵃ/Zᵃ。overlap 模式下的拼接、softmax 归一化和加权求和完整实现了式(11)(12)。

此设计用统一的缓存机制同时支持训练时的一次性分组和推理时的增量压缩，保证了训练与部署行为的一致性。
