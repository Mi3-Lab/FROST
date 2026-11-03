"""ModelOpt PTQ of the velocity field (FieldGraph) to FP8 or NVFP4 and export to a TensorRT-ready ONNX. Workstation.

  python quantize_export.py --mode fp8   --calib dump_a.pt dump_b.pt ... --out DIR
  python quantize_export.py --mode nvfp4 --calib ... --out DIR
  python quantize_export.py --mode fp8 --synthetic --out DIR      (pipeline smoke test only: random prefix cache)

Quantized: every nn.Linear of the expert (q/k/v/o, gate/up/down x 36 layers), weights and inputs, with ModelOpt's
default recipes (FP8_DEFAULT_CFG: E4M3 per-tensor weight and activation scales, activation amax from calibration;
NVFP4_DEFAULT_CFG: E2M1 with E4M3 scales per block of 16 plus a per-tensor scale, activations quantized dynamically
per block at run time). action_in_proj and action_out_proj stay bf16 (as in the registered P1/P3). Attention
(QK^T, PV), norms and the residual stream stay as in the bf16 graph.
Calibration forward: for every dump (one clip each, see dump_clip.py) and every seed, the bf16 Euler-10 path
x_0 -> x_1 is replayed and each (x_t, t) fed to the model, so activation ranges cover the states the solvers visit.
Clips used for calibration must not be evaluation clips.
Export: modelopt.torch._deploy.utils.get_onnx_bytes_and_metadata (Q/DQ for FP8; FP4 weight packing + dynamic
activation quantization for NVFP4), then one external-data file; output renamed to 'v'.
"""
import argparse, copy, json, time
from pathlib import Path
import torch

from field_trt import FieldGraph, load_field_modules, NEG


def prefixes(a, lmax, hkv=8, d=128, layers=36):
    if a.synthetic:
        g = torch.Generator().manual_seed(0); P = 3500
        kv = torch.zeros(layers, 2, hkv, lmax, d, dtype=torch.bfloat16)
        kv[..., :P, :] = (torch.randn(layers, 2, hkv, P, d, generator=g)).to(torch.bfloat16)
        bias = torch.full((lmax,), NEG); bias[:P] = 0
        pos = torch.arange(64) + P
        inv = 1.0 / (5_000_000 ** (torch.arange(0, d, 2).float() / d)); f = pos[:, None].float() * inv[None]
        emb = torch.cat([f, f], -1)
        yield {'kv': kv, 'bias': bias, 'cos': emb.cos(), 'sin': emb.sin(), 'initial': [torch.randn(1, 64, 2, generator=g) for _ in range(3)]}
        return
    for p in a.calib:
        D = torch.load(p, weights_only=False); P = D['P']
        kv = torch.zeros(*D['past_kv'].shape[:3], lmax, D['past_kv'].shape[-1], dtype=torch.bfloat16); kv[..., :P, :] = D['past_kv']
        bias = torch.full((lmax,), NEG); bias[:P] = D['key_bias']
        yield {'kv': kv, 'bias': bias, 'cos': D['cos'], 'sin': D['sin'], 'initial': [D['initial'][s] for s in sorted(D['initial'])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['fp8', 'nvfp4'], required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--calib', type=Path, nargs='*', default=[]); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--lmax', type=int, default=4096); ap.add_argument('--attn', choices=['fused', 'f32'], default='fused')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--disable', nargs='*', default=[], help='extra quantizer-name patterns kept in bf16, e.g. "*layers.0.*"')
    ap.add_argument('--cfg', default=None, help='ModelOpt config name overriding the --mode default (e.g. NVFP4_AWQ_LITE_CFG, NVFP4_MLP_ONLY_CFG)')
    ap.add_argument('--amax', type=Path, default=None, help='calibrated amax buffers from calib_amax.py (skips calibration;'
                    ' one forward initialises the quantizers, then every *_amax buffer is overwritten)')
    ap.add_argument('--save-amax', type=Path, default=None, help='after calibration, save the *_amax buffers here')
    ap.add_argument('--save-mto', type=Path, default=None, help='after calibration, modelopt.torch.opt.save(model) (full quantized state)')
    ap.add_argument('--restore-mto', type=Path, default=None, help='modelopt.torch.opt.restore a --save-mto file instead of calibrating')
    ap.add_argument('--calib-only', action='store_true', help='calibrate (+ --save-amax), skip the ONNX export')
    a = ap.parse_args(); assert a.calib or a.synthetic
    import modelopt.torch.quantization as mtq
    t0 = time.time(); dev = 'cuda'
    expert, in_proj, out_proj, _, _ = load_field_modules(device=dev)
    g = FieldGraph(expert, in_proj, out_proj, attn=a.attn).eval()
    cfg = copy.deepcopy(getattr(mtq, a.cfg) if a.cfg else (mtq.FP8_DEFAULT_CFG if a.mode == 'fp8' else mtq.NVFP4_DEFAULT_CFG))
    cfg['quant_cfg'] += [{'quantizer_name': '*in_proj*', 'enable': False}, {'quantizer_name': '*out_proj*', 'enable': False}]
    cfg['quant_cfg'] += [{'quantizer_name': p, 'enable': False} for p in a.disable]
    data = list(prefixes(a, a.lmax))

    def forward_loop(model):
        with torch.no_grad():
            for c in data:
                kv, bias, cos, sin = c['kv'].to(dev), c['bias'].to(dev), c['cos'].to(dev), c['sin'].to(dev)
                for x0 in c['initial']:
                    x = x0.to(dev)
                    for i in range(a.steps):
                        t = torch.full((1, 1, 1), i / a.steps, device=dev)
                        x = x + model(x, t, kv, bias, cos, sin) / a.steps
                del kv
    c = data[0]; args = (c['initial'][0].to(dev), torch.full((1, 1, 1), .3, device=dev), c['kv'].to(dev), c['bias'].to(dev), c['cos'].to(dev), c['sin'].to(dev))
    with torch.no_grad(): vb = g(*args)                      # bf16 reference before quantization
    import modelopt.torch.opt as mto
    if a.restore_mto is not None:
        g = mto.restore(g, str(a.restore_mto)); print(f'restored quantized state from {a.restore_mto}', flush=True)
    elif a.amax is not None:
        saved = torch.load(a.amax)
        g = mtq.quantize(g, cfg, lambda m: m(*args))
        bufs = dict(g.named_buffers()); n = 0
        for k, v in saved.items():
            assert k in bufs, k
            bufs[k].copy_(v.to(bufs[k].device, bufs[k].dtype)); n += 1
        missing = [k for k in bufs if k.endswith('_amax') and k not in saved]
        assert not missing, missing[:5]
        print(f'loaded {n} amax buffers from {a.amax}', flush=True)
    else:
        g = mtq.quantize(g, cfg, forward_loop)
    if a.save_mto is not None: mto.save(g, str(a.save_mto))
    if a.save_amax is not None:
        torch.save({k: v.detach().cpu() for k, v in g.named_buffers() if k.endswith('_amax')}, a.save_amax)
    qsum = {n: str(m) for n, m in g.named_modules() if n.endswith('layers.0.mlp.down_proj.input_quantizer') or n.endswith('layers.0.self_attn.q_proj.weight_quantizer')}
    # bf16 vs fake-quant field on the first calibration clip (fake quant in torch; the engine is checked on Thor)
    with torch.no_grad(): vq = g(*args)
    fq = {'rel_l2_fakequant_vs_bf16': float((vq - vb).norm() / vb.norm()), 'max_abs': float((vq - vb).abs().max())}
    print(json.dumps(fq), flush=True)
    if a.calib_only:
        (a.out.parent / (a.out.name + '_calib.json')).write_text(json.dumps({'fakequant_check': fq, 'quantizers_example': qsum,
            'calib': [str(p) for p in a.calib], 'seconds': time.time() - t0}, indent=1)); return
    from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata, OnnxBytes
    # ModelOpt's FP8 post-process also inserts FP8 Q/DQ on every Softmax feeding a MatMul (FP8 attention probabilities).
    # Not wanted: P1 quantizes only the Linear layers, attention stays bf16. (It also fails on bf16 graphs.)
    from modelopt.onnx.export.fp8_exporter import FP8QuantExporter
    FP8QuantExporter._insert_qdq_after_softmax = staticmethod(lambda graph: 0)
    # ModelOpt's remove_redundant_casts drops the fp32 Casts in front of the RMSNorm Pow nodes when it believes the
    # preceding MatMul is fp32 (NVFP4 weight DQ with an fp32 global scale). After nvfp4_fix.py makes that DQ bf16, those
    # Casts are needed again (TRT strongly typed: 'POW must have same input types'). Keep the graph's casts as exported.
    import modelopt.torch._deploy.utils.torch_onnx as _tox
    _tox.remove_redundant_casts = lambda graph, *args, **kw: graph
    t1 = time.time()
    dyn = {'x': {0: 'B'}, 't': {0: 'B'}, 'past_kv': {3: 'L'}, 'key_bias': {0: 'L'}}
    b, _ = get_onnx_bytes_and_metadata(g, args, model_name='field', dynamic_axes=dyn, weights_dtype='fp32')
    raw = a.out.parent / (a.out.name + '_raw'); OnnxBytes.from_bytes(b).write_to_disk(str(raw), clean_dir=True); del b
    import onnx
    m = onnx.load(str(raw / 'field.onnx'), load_external_data=True)
    outs = [o.name for o in m.graph.output]; assert len(outs) == 1, outs
    for n in m.graph.node:
        n.output[:] = ['v' if o == outs[0] else o for o in n.output]
    m.graph.output[0].name = 'v'
    a.out.mkdir(parents=True, exist_ok=True)
    onnx.save_model(m, str(a.out / 'field.onnx'), save_as_external_data=True, all_tensors_to_one_file=True, location='field.onnx.data', size_threshold=1024)
    import collections, shutil; shutil.rmtree(raw)
    ops = dict(collections.Counter(n.op_type for n in m.graph.node).most_common())
    meta = {'mode': a.mode, 'cfg': a.cfg or 'default', 'disable': a.disable, 'attn': a.attn, 'calib': [str(p) for p in a.calib], 'synthetic': a.synthetic, 'lmax_calib': a.lmax,
            'fakequant_check': fq, 'quantizers_example': qsum, 'ops': ops,
            'inputs': [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
            'opset': [(o.domain, o.version) for o in m.opset_import], 'quantize_s': t1 - t0, 'export_s': time.time() - t1,
            'files': {p.name: p.stat().st_size for p in a.out.iterdir()}}
    (a.out / 'export_meta.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
