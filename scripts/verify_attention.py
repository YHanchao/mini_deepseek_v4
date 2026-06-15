"""验证 Attention / Compressor / Indexer 在不同 compress_ratio 下的行为"""

import torch
import sys
sys.path.insert(0, "src")

from src.model import RotaryPositionalEmbedding
from src.deepseek import DSArgs, Compressor, Indexer, Attention

DEVICE = "cuda:0"


def make_args(compress_ratio: int) -> DSArgs:
    return DSArgs(
        device=DEVICE,
        dtype=torch.float32,
        d_model=512,
        head_dim=128,
        index_head_dim=64,
        rope_head_dim=32,
        attn_rank=64,
        index_num=4,
        output_group=4,
        output_lora=512,
        n_heads=8,
        window_size=8,
        compress_ratio=compress_ratio,
        max_batch_len=2,
        max_seq_len=256,
    )


def make_rope(args: DSArgs) -> RotaryPositionalEmbedding:
    return RotaryPositionalEmbedding(
        theta=10000.0, d_k=args.rope_head_dim, max_seq_len=args.max_seq_len, device=DEVICE
    )


def check(condition, msg):
    if not condition:
        print(f"  FAIL: {msg}")
        return False
    print(f"  OK: {msg}")
    return True


# ---------------------------------------------------------------------------
# Test 1: _get_window_topk_id  &  _get_compress_topk_id
# ---------------------------------------------------------------------------
def test_topk_methods():
    print("\n=== Test 1: _get_window_topk_id & _get_compress_topk_id ===")
    args = make_args(4)
    rope = make_rope(args)
    attn = Attention(args, rope)
    batch, seq_len = 2, 12

    # --- Window topk, prefill ---
    win = attn._get_window_topk_id(batch, seq_len, start_pos=0)
    ok = True
    ok &= check(win.shape == (batch, seq_len, args.window_size), f"win prefill shape {win.shape}")
    ok &= check((win[0, 0, :] == -1).sum().item() == 7, "row 0: 7 invalid")
    ok &= check((win[0, 4, :] == -1).sum().item() == 3, "row 4: 3 invalid")
    ok &= check((win[0, 11, :] == -1).sum().item() == 0, "row 11: 0 invalid")
    ok &= check((win[0, 11, 0] == 4).item(), "last row starts at 4")

    # --- Window topk, decode ---
    win_dec = attn._get_window_topk_id(batch, seq_len=1, start_pos=10)
    ok &= check(win_dec.shape == (batch, 1, args.window_size), f"win decode shape {win_dec.shape}")

    # --- Compress topk, prefill (ratio=4, seq_len=12) ---
    # 因果掩码: t=0..2 看不见任何压缩块(-1), t=3..6 看到块0, t=7..10 看到块0&1, t=11 看到块0&1&2
    comp = attn._get_compress_topk_id(batch, seq_len, start_pos=0, offset=0)
    n_blocks = seq_len // args.compress_ratio  # 3
    ok &= check(comp.shape == (batch, seq_len, n_blocks), f"comp prefill shape {comp.shape}")
    ok &= check((comp[0, 0, :] == -1).all().item(), "t=0: all -1")
    ok &= check((comp[0, 2, :] == -1).all().item(), "t=2: all -1")
    ok &= check((comp[0, 3, 0] == 0).item() and (comp[0, 3, 1] == -1).item(), "t=3: [0,-1,-1]")
    ok &= check((comp[0, 4, 0] == 0).item() and (comp[0, 4, 1] == -1).item(), "t=4: [0,-1,-1]")
    ok &= check((comp[0, 7, 0] == 0).item() and (comp[0, 7, 1] == 1).item(), "t=7: [0,1,-1]")
    ok &= check((comp[0, 11, 0] == 0).item() and (comp[0, 11, 2] == 2).item(), "t=11: [0,1,2]")

    # --- Compress topk, decode ---
    comp_dec = attn._get_compress_topk_id(batch, seq_len=1, start_pos=10, offset=8)
    n_blocks_dec = (10 + 1) // 4  # 2
    ok &= check(comp_dec.shape == (batch, 1, n_blocks_dec), f"comp decode shape {comp_dec.shape}")
    ok &= check((comp_dec[0, 0, 0] == 8).item(), "decode: first block offset=8")
    ok &= check((comp_dec[0, 0, 1] == 9).item(), "decode: second block offset=9")

    return ok


# ---------------------------------------------------------------------------
# Test 2: Compressor with ratio=4 (overlap)
# ---------------------------------------------------------------------------
def test_compressor_overlap():
    print("\n=== Test 2: Compressor ratio=4 (overlap) ===")
    args = make_args(4)
    rope = make_rope(args)
    compressor = Compressor(args, head_dim=128, compress_ratio=4, rope=rope)

    ok = True
    ok &= check(compressor.overlap == True, "overlap flag")
    # coff=2 → weight_kv output is 2*head_dim=256
    ok &= check(compressor.weight_kv.weight.shape == (256, 512), f"wkv shape {compressor.weight_kv.weight.shape}")
    ok &= check(compressor.bias.shape == (4, 256), f"bias shape {compressor.bias.shape}")
    # coff=2 → kv_state is (batch, 2*4=8, 2*128=256)
    ok &= check(compressor.kv_state.shape == (2, 8, 256), f"kv_state shape {compressor.kv_state.shape}")

    compressor.kv_cache = torch.zeros(2, 64, 128, device=DEVICE)

    # Prefill: 10 tokens → 2 compressed blocks (8 tokens) + 2 remainder
    x = torch.randn(2, 10, 512, device=DEVICE)
    out = compressor(x, start_pos=0)
    ok &= check(out is not None, "prefill returned tensor")
    ok &= check(out.shape == (2, 2, 128), f"prefill out shape {out.shape}")

    # Decode: positions 10, 11 → both stored, 11 triggers 3rd compression
    for sp in [10, 11]:
        xd = torch.randn(2, 1, 512, device=DEVICE)
        out = compressor(xd, start_pos=sp)
        if (sp + 1) % 4 == 0:
            ok &= check(out is not None, f"decode pos {sp} compressed")
            ok &= check(out.shape == (2, 1, 128), f"decode out shape {out.shape}")
        else:
            ok &= check(out is None, f"decode pos {sp} no compress yet")

    return ok


# ---------------------------------------------------------------------------
# Test 3: Compressor with ratio=128 (non-overlap)
# ---------------------------------------------------------------------------
def test_compressor_non_overlap():
    print("\n=== Test 3: Compressor ratio=128 (non-overlap) ===")
    args = make_args(128)
    rope = make_rope(args)
    compressor = Compressor(args, head_dim=128, compress_ratio=128, rope=rope)

    ok = True
    ok &= check(compressor.overlap == False, "overlap flag is False")
    # coff=1 → weight_kv output is head_dim=128
    ok &= check(compressor.weight_kv.weight.shape == (128, 512), f"wkv shape {compressor.weight_kv.weight.shape}")
    ok &= check(compressor.bias.shape == (128, 128), f"bias shape {compressor.bias.shape}")
    # coff=1 → kv_state is (batch, 128, 128)
    ok &= check(compressor.kv_state.shape == (2, 128, 128), f"kv_state shape {compressor.kv_state.shape}")

    compressor.kv_cache = torch.zeros(2, 2, 128, device=DEVICE)

    # Prefill: 200 tokens → 1 compressed block (128 tokens) + 72 remainder
    x = torch.randn(2, 200, 512, device=DEVICE)
    out = compressor(x, start_pos=0)
    ok &= check(out is not None, "prefill returned tensor")
    ok &= check(out.shape == (2, 1, 128), f"prefill out shape {out.shape}")

    # Decode: fill up to ratio=128
    # After prefill: kv_state has 72 remainder at [0:72]
    # Need 128-72=56 more decode tokens → compression at start_pos=200+55=255
    for sp in range(200, 255):
        xd = torch.randn(2, 1, 512, device=DEVICE)
        out = compressor(xd, start_pos=sp)
    ok &= check(out is None, "no compress at pos 254")

    xd = torch.randn(2, 1, 512, device=DEVICE)
    out = compressor(xd, start_pos=255)
    ok &= check(out is not None, "compress at pos 255")
    ok &= check(out.shape == (2, 1, 128), f"decode out shape {out.shape}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Indexer
# ---------------------------------------------------------------------------
def test_indexer():
    print("\n=== Test 4: Indexer ===")
    args = make_args(4)
    rope = make_rope(args)
    indexer = Indexer(args, rope)

    indexer.kv_cache = torch.zeros(2, 64, args.index_head_dim, device=DEVICE)
    indexer.compressor.kv_cache = indexer.kv_cache

    ok = True
    ok &= check(indexer.kv_cache.shape == (2, 64, args.index_head_dim), f"kv_cache shape {indexer.kv_cache.shape}")
    ok &= check(indexer.compressor.overlap == True, "indexer compressor overlap=True")

    # Prefill
    x = torch.randn(2, 10, 512, device=DEVICE)
    qr = torch.randn(2, 10, args.attn_rank, device=DEVICE)
    idxs = indexer(x, qr, start_pos=0, offset=10)
    # topk = min(index_topk=4, end_pos//ratio=2) = 2 (只有2个压缩块)
    ok &= check(idxs.shape == (2, 10, 2), f"prefill idxs shape {idxs.shape}")

    # Decode
    xd = torch.randn(2, 1, 512, device=DEVICE)
    qrd = torch.randn(2, 1, args.attn_rank, device=DEVICE)
    idxs_dec = indexer(xd, qrd, start_pos=10, offset=8)
    ok &= check(idxs_dec.shape == (2, 1, 2), f"decode idxs shape {idxs_dec.shape}")

    return ok


# ---------------------------------------------------------------------------
# Test 5: Attention with ratio=4
# ---------------------------------------------------------------------------
def test_attention_ratio4():
    print("\n=== Test 5: Attention ratio=4 (full end-to-end) ===")
    args = make_args(4)
    rope = make_rope(args)
    attn = Attention(args, rope)

    ok = True
    ok &= check(attn.compressor is not None, "compressor created")
    ok &= check(attn.indexer is not None, "indexer created")
    ok &= check(attn.kv_cache_size == 8 + 256 // 4, f"kv_cache_size {attn.kv_cache_size}")

    # Prefill
    x = torch.randn(2, 12, 512, device=DEVICE)
    out = attn(x, start_pos=0)
    ok &= check(out.shape == (2, 12, 512), f"prefill output shape {out.shape}")

    # Decode
    for sp in range(12, 15):
        xd = torch.randn(2, 1, 512, device=DEVICE)
        out = attn(xd, start_pos=sp)
        ok &= check(out.shape == (2, 1, 512), f"decode pos {sp} shape {out.shape}")

    return ok


# ---------------------------------------------------------------------------
# Test 6: Attention with ratio=128
# ---------------------------------------------------------------------------
def test_attention_ratio128():
    print("\n=== Test 6: Attention ratio=128 ===")
    args = make_args(128)
    rope = make_rope(args)
    attn = Attention(args, rope)

    ok = True
    ok &= check(attn.compressor is not None, "compressor created")
    ok &= check(attn.indexer is None, "indexer is None (no overlap)")
    ok &= check(attn.kv_cache_size == 8 + 256 // 128, f"kv_cache_size {attn.kv_cache_size}")

    # Prefill
    x = torch.randn(2, 200, 512, device=DEVICE)
    out = attn(x, start_pos=0)
    ok &= check(out.shape == (2, 200, 512), f"prefill output shape {out.shape}")

    return ok


# ---------------------------------------------------------------------------
# Test 7: Attention with ratio=0 (pure sliding window)
# ---------------------------------------------------------------------------
def test_attention_ratio0():
    print("\n=== Test 7: Attention ratio=0 (pure sliding window) ===")
    args = make_args(0)
    rope = make_rope(args)
    attn = Attention(args, rope)

    ok = True
    ok &= check(attn.compressor is None, "compressor is None")
    ok &= check(attn.indexer is None, "indexer is None")
    ok &= check(attn.kv_cache_size == 8, f"kv_cache_size {attn.kv_cache_size} (window only)")

    # Prefill
    x = torch.randn(2, 12, 512, device=DEVICE)
    out = attn(x, start_pos=0)
    ok &= check(out.shape == (2, 12, 512), f"prefill output shape {out.shape}")

    # Decode
    for sp in range(12, 15):
        xd = torch.randn(2, 1, 512, device=DEVICE)
        out = attn(xd, start_pos=sp)
        ok &= check(out.shape == (2, 1, 512), f"decode pos {sp} shape {out.shape}")

    return ok


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    results = []
    results.append(test_topk_methods())
    results.append(test_compressor_overlap())
    results.append(test_compressor_non_overlap())
    results.append(test_indexer())
    results.append(test_attention_ratio4())
    results.append(test_attention_ratio128())
    results.append(test_attention_ratio0())

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print(f"FAILED: {total - passed} test(s)")
        sys.exit(1)
