"""Export the Alpamayo-1.5 velocity field (FieldGraph) to ONNX with static shapes. Runs on the workstation.

  python export_onnx.py --out /path/onnx_b1_L4096_fused --batch 1 --lmax 4096 --attn fused [--check cuda]

--check DEVICE first compares FieldGraph against the HF expert exactly as the eager field calls it
(torch.autocast bf16, SDPA, DynamicCache prefix, 4-D float mask, crop) on a synthetic prefix with a masked gap and
padding, and writes check.json. The export itself runs on CPU (bf16 weights, ~4.6 GB of external data).
"""
import argparse, json, time
from pathlib import Path
import torch

from field_trt import FieldGraph, load_field_modules, rope_cos_sin, pack_prefix, NEG


def synthetic_prefix(expert, P, gap, device, seed=0):
    from transformers import DynamicCache
    g = torch.Generator(device='cpu').manual_seed(seed)
    cfg = expert.config; Lr, hkv, d = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    cache = DynamicCache()
    for i in range(Lr):
        k = (torch.randn(1, hkv, P, d, generator=g) * 2).to(torch.bfloat16).to(device)
        v = torch.randn(1, hkv, P, d, generator=g).to(torch.bfloat16).to(device)
        cache.update(k, v, i)
    n = 64
    mask = torch.zeros(1, 1, n, P + n, device=device)
    mask[..., P - gap:P] = NEG                                      # like [offset : -n]
    pos = torch.arange(n, device=device).view(1, 1, n).expand(3, 1, n) + (P - gap)
    return cache, mask, pos


def hf_field(expert, in_proj, out_proj, cache, mask, pos, x, t):
    """The eager field of frost.probes.alpamayo_field.prepare_field (+ batch broadcast of thor_latency.batched_field)."""
    B, n = x.shape[0], 64
    P = cache.get_seq_length()
    base = [(l.keys, l.values) for l in cache.layers]
    for l, (k, v) in zip(cache.layers, base): l.keys, l.values = k.expand(B, *k.shape[1:]), v.expand(B, *v.shape[1:])
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            e = in_proj(x, t)
            r = expert(inputs_embeds=e, position_ids=pos.expand(-1, B, -1), past_key_values=cache,
                       attention_mask=mask.expand(B, -1, -1, -1), use_cache=True, is_causal=False)
            out = out_proj(r.last_hidden_state[:, -n:]).view(B, n, 2).float()
    finally:
        for l, (k, v) in zip(cache.layers, base): l.keys, l.values = k, v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--lmax', type=int, default=4096); ap.add_argument('--attn', choices=['fused', 'f32'], default='fused')
    ap.add_argument('--check', default=None, help='device for the HF-vs-FieldGraph check (e.g. cuda); skipped if unset')
    ap.add_argument('--check-prefix', type=int, default=3500); ap.add_argument('--opset', type=int, default=19)
    ap.add_argument('--no-export', action='store_true')
    ap.add_argument('--static', action='store_true', help='fixed B and L in the ONNX graph (default: dynamic B, L; the engine fixes them)')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(16)
    meta = {'args': {k: str(v) for k, v in vars(a).items()}, 'torch': torch.__version__}

    if a.check:
        dev = a.check
        expert, in_proj, out_proj, aspace, cfg = load_field_modules(device=dev)
        g = FieldGraph(expert, in_proj, out_proj, attn=a.attn).eval()
        cache, mask, pos = synthetic_prefix(expert, a.check_prefix, gap=37, device=dev)
        kv, bias, P = pack_prefix(cache, mask, a.lmax)
        cos, sin = rope_cos_sin(expert, pos)
        res = {}
        for B in sorted({1, a.batch}):
            torch.manual_seed(1)
            x = torch.randn(B, 64, 2, device=dev); errs = []
            for tv in [0.0, 0.3, 0.9]:
                t = torch.full((B, 1, 1), tv, device=dev)
                ref = hf_field(expert, in_proj, out_proj, cache, mask, pos, x, t)
                with torch.inference_mode(): got = g(x, t, kv, bias, cos, sin)
                errs.append({'t': tv, 'max_abs': float((got - ref).abs().max()), 'ref_abs_max': float(ref.abs().max()),
                             'rel_l2': float((got - ref).norm() / ref.norm())})
            res[f'B{B}'] = errs
        # freqs buffers (bf16 when built like from_pretrained) for the cross-check against the Thor model
        res['in_proj_freqs_dtype'] = str(in_proj.sinus[0].freqs.dtype)
        res['in_proj_freqs'] = in_proj.sinus[0].freqs.float().flatten().tolist()
        meta['check'] = res; print(json.dumps({k: v for k, v in res.items() if k != 'in_proj_freqs'}, indent=1))
        (a.out / 'check.json').write_text(json.dumps(res, indent=1))
        del expert, in_proj, out_proj, g, cache, kv; torch.cuda.empty_cache()
    if a.no_export: return

    t0 = time.time()
    expert, in_proj, out_proj, aspace, cfg = load_field_modules(device='cpu')
    g = FieldGraph(expert, in_proj, out_proj, attn=a.attn).eval()
    cfgx = expert.config; Lr, hkv, d = cfgx.num_hidden_layers, cfgx.num_key_value_heads, cfgx.head_dim
    B, L = a.batch, a.lmax
    args = (torch.randn(B, 64, 2), torch.rand(B, 1, 1), torch.randn(Lr, 2, hkv, L, d).to(torch.bfloat16),
            torch.zeros(L), torch.randn(64, d), torch.randn(64, d))
    path = a.out / 'field.onnx'
    with torch.inference_mode():
        torch.onnx.export(g, args, str(path), input_names=['x', 't', 'past_kv', 'key_bias', 'cos', 'sin'],
                          output_names=['v'], opset_version=a.opset, dynamo=False, external_data=True, do_constant_folding=True,
                          dynamic_axes=None if a.static else {'x': {0: 'B'}, 't': {0: 'B'}, 'v': {0: 'B'}, 'past_kv': {3: 'L'}, 'key_bias': {0: 'L'}})
    meta.update({'export_s': time.time() - t0, 'shapes': {'x': [B, 64, 2], 't': [B, 1, 1], 'past_kv': [Lr, 2, hkv, L, d],
                 'key_bias': [L], 'cos': [64, d], 'sin': [64, d], 'v': [B, 64, 2]},
                 'files': {p.name: p.stat().st_size for p in a.out.iterdir()}})
    (a.out / 'export_meta.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta['files'], indent=1))


if __name__ == '__main__':
    main()
