# 内存泄漏排查记录

## 当前状态

PyTorch 2.8 镜像，5×4090 DDP，small 配置（305M），seq=1024，bs=4。
训练已跑通（能出 checkpoint），但 CPU 内存线性上涨。
9 小时实验已完成：2.8 同样 OOM。

## 实验结论

| 实验 | 结果 | 结论 |
|------|------|------|
| `retain_graph=True` 双 backward（初版） | 9h → 256GB OOM | C++ autograd 图不释放 |
| 删除 `retain_graph`，合成单 `total_loss.backward()` | 仍然 OOM | `retain_graph` 不是唯一元凶 |
| `find_unused_parameters=False` | 启动即报错 | 有结构性不可达参数 |
| `0.0 * p.sum()` 试图绕过 | 不可达参数更多了 | 零系数被 autograd 优化掉 |
| PyTorch 2.4 / 2.8 均泄漏 | 线性上涨无变化 | **泄漏与 DDP 版本无关** |
| 4（或5）个进程内存曲线一致 | 每个进程 ~25% 系统内存 | 泄漏源在每进程共同调用的代码路径

## 核心代码改动（已合入）

路径：`src/trainer.py`

执行所有改动后仍然存在内存泄漏问题。

### `train_step` 改动

```python
# 旧版：两次 backward，retain_graph
idx_grads = torch.autograd.grad(kl_loss_scaled, idx_p, retain_graph=True, allow_unused=True)
for p, g in zip(idx_p, idx_grads):
    if g is not None:
        p.grad = g if p.grad is None else p.grad.add_(g)
lm_loss_scaled.backward()

# 新版：一次 backward，wc.detach() 切断 attention → KL 梯度
kl_loss = sum(
    indexer_kl_loss(iscore, idx, wc.detach() if wc is not None else None)
    for (iscore, wc, idx) in idx_data
)
total_loss = (lm_loss + 0.5 * kl_loss) / self.grad_accum
total_loss.backward()
```

### `build_model_and_optimizers` 改动

`find_unused_parameters=True`（不变）——有结构性不可达参数（indexer 在无 compress_ratio 的层），必须 True。

## 可能的排查路线

### 嫌疑人 #1：pin_memory + non_blocking

当前 `DataLoader(pin_memory=True)` + `input_ids.to(device, non_blocking=True)`。
pinned memory 不会被 OS 回收，且 non_blocking 的异步传输期间 CPU tensor 不被释放。
DataLoader 的 pin_memory 分配器和 Python GC 打架是 PyTorch 老问题。

**验证**：`trainer.py` 中把 `pin_memory=True` 改成 `False`，跑 500 步看内存。

### 嫌疑人 #2：DistributedSampler + mmap

4（或5）个进程各自 mmap 同一个 6.2G 文件，内核 page cache 复制多份。
虽然 MADV_SEQUENTIAL 设置了，但 DDP 各个 rank 的采样 stride=world_size，
不是严格顺序访问，内核可能为每个进程单独缓存。

**仓库持有者的注释**：但这仍然不能解释为何 256 GB CPU 内存会被耗尽。即使每个 rank 把 6.2G 文件全部读入内存，那么充其量不会超过 40G，而目前内存会持续线性上涨。所以我不认为是这个原因

**验证**：用 `while true; do echo 1 > /proc/sys/vm/drop_caches; sleep 60; done` 定时清 page cache，看 RSS 是否回落。

### 嫌疑人 #3：mHC checkpoint 引用环

ManifoldHyperConnections 使用 `torch.utils.checkpoint.checkpoint`，
该方法在复杂模型中已知会产生 Python 侧引用环。
每个 `checkpoint` 调用创建 recompute_context，持有 saved activations 的引用。

**仓库持有者的注释**：我不认为是这个原因。我理解现在一大堆问题都是由于 Indexer 导致的——为了学习 Indexer，我必须得维护两个计算图。但 mHC 本身是在主干 LLM 上的，我不认为是因为 checkpoint 的缘故。更何况，checkpoint 应当影响 GPU 内存，而显存始终保持正常，没有出现内存泄漏。

**验证**：`DSArgs(use_checkpoint=False)` 跑 500 步对比内存曲线。

### 嫌疑人 #4：autograd 图 Python wrapper 堆积

单 backward 虽然释放了 C++ 侧图节点，但 Python tensor wrapper 可能靠 GC 才回收。
每步数百万 tensor wrapper，GC 阈值（默认 700）触发不及时。
当前每 100 步才调一套 `gc.collect()`。

**仓库持有者的注释**：我不认为是这个原因。因为9个小时运行了25000步，那么理论上会触发250次 GC。如果GC确实有效，那么应当看到内存会呈现锯齿状上升又回落。但事实上内存仍然以线性形式上涨，所以一定是某个动态创建的变量，其 Python 引用数不为 0，压根不会被 GC 回收。

**验证**：把 `step % 100` 改成 `step % 1`，每步强制 GC。

## 快速消融实验脚本

```bash
# 单卡验证（不需要 DDP，在 DGX 上跑也行）
# 把 PYTHONTRACEMALLOC=10 替换成你想要的实验
cd /mnt/MiniDSv4
for test in no_pin_memory no_checkpoint gc_every_step; do
    echo "=== Testing: $test ==="
    PYTHONPATH=. timeout 600 python scripts/pretrain.py \
        --config-name tiny \
        --data-train data/TinyStoriesV2_valid.bin \
        --total-steps 500 \
        --log-every 50 \
        --wandb-project "" \
        --output-dir /tmp/leak_${test}
done
```

## 兜底方案

如果上述全部无法定位，用 `timeout` 包裹训练命令，每 N 小时自动 kill + resume：

```bash
while true; do
    newest=$(ls -t checkpoints/pretrain/ckpt_*.pt 2>/dev/null | grep -v nan | head -1)
    resume_arg=""
    [ -n "$newest" ] && resume_arg="--resume $newest"
    timeout 4h torchrun --nproc_per_node=5 scripts/pretrain.py \
        --data-train data/train.bin \
        --data-val data/valid.bin \
        --total-steps 401000 \
        --warmup-steps 0 \
        --log-every 10 \
        $resume_arg
    echo "$(date): restarting..."
    sleep 5
done
```

注意：resume 时 warmup 会重新走——改为 `--warmup-steps 0`，或者修改 `PretrainTrainer.get_lr()` 让 warmup 基于 resume 后的续训步数而非从 0 开始。

**仓库持有者的注释**：我**强烈不认同**这个方案，我非常不认同把这个问题隐藏起来。未来我还要在这个基础上做后训练，总不能全靠定时 kill 来解决吧？
