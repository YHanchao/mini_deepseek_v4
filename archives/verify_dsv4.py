"""验证 DeepSeekV4 模型推理正确性"""

import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import RotaryPositionalEmbedding
from src.deepseek import DSArgs, Compressor, Indexer, Attention, DeepSeekV4

DEVICE = "cuda:0"
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(DEVICE)
torch.manual_seed(42)


def check(condition, msg):
    if not condition:
        print(f"  FAIL: {msg}")
        return False
    print(f"  OK: {msg}")
    return True


# ---------------------------------------------------------------------------
# Test 1: 基础组件 shape 检查
# ---------------------------------------------------------------------------
def test_shapes():
    print("\n=== Test 1: 基础组件 shape ===")
    args = DSArgs(compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0), n_hash_layer=0)
    rope = RotaryPositionalEmbedding(10000.0, args.rope_head_dim, args.max_seq_len, device=DEVICE)

    # Compressor ratio=4 (overlap)
    c4 = Compressor(args, head_dim=256, compress_ratio=4, rope=rope)
    ok = check(c4.overlap == True, "compressor overlap=True")
    ok &= check(c4.weight_kv.weight.shape == (512, 2048), f"wkv {c4.weight_kv.weight.shape}")
    ok &= check(c4.kv_state.shape == (4, 8, 512), f"kv_state {c4.kv_state.shape}")

    # Compressor ratio=128 (non-overlap)
    c128 = Compressor(args, head_dim=256, compress_ratio=128, rope=rope)
    ok &= check(c128.overlap == False, "compressor overlap=False")
    ok &= check(c128.weight_kv.weight.shape == (256, 2048), f"wkv {c128.weight_kv.weight.shape}")
    ok &= check(c128.kv_state.shape == (4, 128, 256), f"kv_state {c128.kv_state.shape}")

    # Attention ratio=0
    a0 = Attention(args, rope, layer_id=0)
    ok &= check(a0.compress_ratio == 0, "layer 0 ratio=0")
    ok &= check(a0.compressor is None, "layer 0 no compressor")
    ok &= check(a0.indexer is None, "layer 0 no indexer")
    ok &= check(a0.kv_cache_size == 128, f"kv_cache_size {a0.kv_cache_size}")

    # Attention ratio=4
    a4 = Attention(args, rope, layer_id=2)
    ok &= check(a4.compress_ratio == 4, "layer 2 ratio=4")
    ok &= check(a4.compressor is not None, "layer 2 has compressor")
    ok &= check(a4.indexer is not None, "layer 2 has indexer")

    # Attention ratio=128
    a128 = Attention(args, rope, layer_id=3)
    ok &= check(a128.compress_ratio == 128, "layer 3 ratio=128")
    ok &= check(a128.compressor is not None, "layer 3 has compressor")
    ok &= check(a128.indexer is None, "layer 3 no indexer")

    return ok


# ---------------------------------------------------------------------------
# Test 2: Topk indices
# ---------------------------------------------------------------------------
def test_topk():
    print("\n=== Test 2: topk indices ===")
    args = DSArgs(compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0), n_hash_layer=0)
    rope = RotaryPositionalEmbedding(10000.0, args.rope_head_dim, args.max_seq_len, device=DEVICE)
    attn = Attention(args, rope, layer_id=2)
    b, s = 2, 16

    win = attn._get_window_topk_id(b, s, start_pos=0)
    # seq_len=16 < window_size=128, 所以窗口大小被截断为 seq_len
    ok = check(win.shape == (b, s, min(s, 128)), f"win prefill {win.shape}")
    ok &= check((win[0, 0, 1:] == -1).all().item(), "row 0: only first valid")
    ok &= check((win[0, 15, 0] >= 0).item(), "row 15: all valid")

    win_d = attn._get_window_topk_id(b, 1, start_pos=10)
    ok &= check(win_d.shape == (b, 1, 128), f"win decode {win_d.shape}")

    comp = attn._get_compress_topk_id(b, s, start_pos=0, offset=0)
    ok &= check(comp.shape == (b, s, s // 4), f"comp prefill {comp.shape}")
    ok &= check((comp[0, 0, :] == -1).all().item(), "row 0: all masked")
    ok &= check((comp[0, 3, 0] == 0).item(), "row 3: sees block 0")

    comp_d = attn._get_compress_topk_id(b, 1, start_pos=10, offset=16)
    ok &= check(comp_d.shape == (b, 1, (10 + 1) // 4), f"comp decode {comp_d.shape}")
    return ok


# ---------------------------------------------------------------------------
# Test 3: Compressor forward
# ---------------------------------------------------------------------------
def test_compressor():
    print("\n=== Test 3: Compressor forward ===")
    args = DSArgs(compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0), n_hash_layer=0)
    rope = RotaryPositionalEmbedding(10000.0, args.rope_head_dim, args.max_seq_len, device=DEVICE)

    # overlap (ratio=4): seqlen=30 → 7 blocks + 2 remainder
    c = Compressor(args, head_dim=256, compress_ratio=4, rope=rope)
    c.kv_cache = torch.zeros(4, 512, 256)
    x = torch.randn(2, 30, 2048)
    out = c(x, start_pos=0)
    ok = check(out is not None, "prefill ratio=4 returns tensor")
    ok &= check(out.shape == (2, 7, 256), f"prefill out {out.shape}")

    # Decode
    for sp in [30, 31]:
        xd = torch.randn(2, 1, 2048)
        out = c(xd, start_pos=sp)
        if (sp + 1) % 4 == 0:
            ok &= check(out is not None, f"decode pos {sp} compressed")
        else:
            ok &= check(out is None, f"decode pos {sp} no compress")

    # non-overlap (ratio=128): seqlen=200 → 1 block + 72 remainder
    c2 = Compressor(args, head_dim=256, compress_ratio=128, rope=rope)
    c2.kv_cache = torch.zeros(4, 16, 256)
    x2 = torch.randn(2, 200, 2048)
    out2 = c2(x2, start_pos=0)
    ok &= check(out2 is not None, "prefill ratio=128 returns tensor")
    ok &= check(out2.shape == (2, 1, 256), f"prefill out {out2.shape}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: 全模型 Prefill + Decode
# ---------------------------------------------------------------------------
def test_full_model():
    print("\n=== Test 4: 全模型 Prefill ===")
    args = DSArgs(compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0), n_hash_layer=0)
    x = torch.randint(0, args.vocab_size, (2, 64))
    dpsk = DeepSeekV4(args)

    ntp, mtp_list = dpsk(x)
    ok = check(ntp.shape == (2, 64, args.vocab_size), f"ntp {ntp.shape}")
    ok &= check(len(mtp_list) == args.n_mtp_layer, f"mtp layers {len(mtp_list)}")
    ok &= check(mtp_list[0].shape == (2, 64, args.vocab_size), f"mtp[0] {mtp_list[0].shape}")

    print("\n=== Test 4b: 全模型 Decode ===")
    for pos in range(64, 68):
        ntp, _ = dpsk(x[:, 0:1], pos)
        ok &= check(ntp.shape == (2, 1, args.vocab_size), f"pos {pos} ntp {ntp.shape}")
    return ok


# ---------------------------------------------------------------------------
# Test 5: MTP 链式调用
# ---------------------------------------------------------------------------
def test_mtp():
    print("\n=== Test 5: MTP ===")
    args = DSArgs(compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0), n_hash_layer=0)
    x = torch.randint(0, args.vocab_size, (2, 64))
    dpsk = DeepSeekV4(args)

    h = torch.randn(2, 64, args.expansion_rate, args.d_model)
    mtp = dpsk.mtp_layers[0]
    logits, hidden = mtp(h, start_pos=0, token_ids=x)
    ok = check(logits.shape == (2, 64, args.vocab_size), f"mtp logits {logits.shape}")
    ok &= check(hidden.shape == (2, 64, args.expansion_rate, args.d_model), f"mtp hidden {hidden.shape}")

    logits2, _ = mtp(h[:, 0:1], start_pos=1, token_ids=x[:, 0:1])
    ok &= check(logits2.shape == (2, 1, args.vocab_size), f"mtp decode {logits2.shape}")
    return ok


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = []
    results.append(test_shapes())
    results.append(test_topk())
    results.append(test_compressor())
    results.append(test_full_model())
    results.append(test_mtp())

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print(f"FAILED: {total - passed} test(s)")
        sys.exit(1)
