"""Export the Alpamayo-1.5 VLM graphs (vlm_graph.py) to ONNX, bf16 or ModelOpt FP8 / NVFP4. Runs on the Thor (GPU
for calibration; the workstation GPU is reserved for other jobs).

  python export_vlm.py bf16      --out $FROST_THOR_ROOT/vlm/onnx/bf16
  python export_vlm.py calibdata --out $FROST_THOR_ROOT/vlm/calib          # 8 dev-set clips (not evaluation clips)
  python export_vlm.py quant     --precision fp8   --calib $FROST_THOR_ROOT/vlm/calib --out $FROST_THOR_ROOT/vlm/onnx/fp8
  python export_vlm.py quant     --precision nvfp4 --calib ... --out $FROST_THOR_ROOT/vlm/onnx/nvfp4 [--disable '*layers.0.*']

Layout: OUT/vision/vision.onnx, OUT/chunk{i}/chunk.onnx (layers [9i, 9i+9)), OUT/head/head.onnx, each with one
external-data file. Quantized exports contain only the chunks; vision and head stay bf16 (build them from the bf16
export). Quantized = every nn.Linear of the 36 decoder layers (q/k/v/o, gate/up/down); embeddings, lm_head, norms,
attention (QK^T, softmax, PV) and the residual stream stay bf16, as in the action-expert engines (deploy/thor/trt).
  FP8  : ModelOpt FP8_DEFAULT_CFG (E4M3, per-tensor static weight and activation scales, max calibration).
  NVFP4: ModelOpt NVFP4_DEFAULT_CFG (E2M1 weights, E4M3 scale per 16-block + fp32 global; activations quantised
         dynamically per block by TensorRT at run time, global activation scale from calibration).
Calibration data (calibdata): the bf16 PyTorch graph rolls out the reasoning on each dev clip (seed 123, same sampling
as the probe); the full sequence prompt + reasoning is then the calibration input, so the scales cover prompt (image
and text) tokens and reasoning tokens. Chunk k is calibrated on the bf16 outputs of chunk k-1 (ModelOpt calibrates
an unquantised forward, so this is the same as calibrating the whole model at once).
ModelOpt 0.47 export workarounds are the same as for the field (deploy/thor/trt/quantize_export.py, nvfp4_fix.py).
"""
import os
import argparse, collections, copy, gc, json, shutil, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trt'))
from vlm_graph import (BF16, NEG, HeadGraph, TextChunkGraph, VisionGraph, load_head, load_text_layers, load_vision)  # noqa: E402

CALIB_DIR = Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/trt/calib_data_testv2')
N_CHUNKS = 4; PPI = 720; N_IMG = 16


def bounds():
    return [(i * 9, i * 9 + 9) for i in range(N_CHUNKS)]


def settle(seconds=5):
    """The board's SSD is a USB T7 that reset once under heavy writes: flush after every multi-GB file and pause."""
    import os; os.sync(); time.sleep(seconds)


def _save_onnx(src_file, dst_dir, out_names=None, name='model'):
    """Re-save with one external-data file (and optionally rename the graph outputs in order)."""
    import onnx
    m = onnx.load(str(src_file), load_external_data=True)
    if out_names:
        old = [o.name for o in m.graph.output]; assert len(old) == len(out_names), old
        ren = dict(zip(old, out_names))
        for n in m.graph.node: n.output[:] = [ren.get(o, o) for o in n.output]
        for o in m.graph.output: o.name = ren[o.name]
    dst_dir.mkdir(parents=True, exist_ok=True)
    onnx.save_model(m, str(dst_dir / f'{name}.onnx'), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=f'{name}.onnx.data', size_threshold=1024)
    io = {'inputs': [(i.name, i.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
          'outputs': [(i.name, i.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.output],
          'ops': dict(collections.Counter(n.op_type for n in m.graph.node).most_common())}
    (dst_dir / 'graph_summary.json').write_text(json.dumps(io, indent=1))
    settle()
    return io


CHUNK_IN = ['h', 'past_kv', 'bias', 'cos', 'sin']
CHUNK_DYN = {'h': {1: 'S'}, 'past_kv': {3: 'C'}, 'bias': {0: 'S', 1: 'CS'}, 'cos': {0: 'S'}, 'sin': {0: 'S'},
             'deepstack': {1: 'S'}, 'h_out': {1: 'S'}, 'new_kv': {3: 'S'}}


def chunk_args(g, S=8, C=4, device='cpu'):
    n = len(g.layers)
    args = [torch.randn(1, S, 4096, dtype=BF16, device=device), torch.randn(n, 2, 8, C, 128, dtype=BF16, device=device),
            torch.zeros(S, C + S, device=device), torch.randn(S, 128, dtype=BF16, device=device), torch.randn(S, 128, dtype=BF16, device=device)]
    if g.has_deepstack: args.append(torch.zeros(3, S, 4096, dtype=BF16, device=device))
    return tuple(args)


def export_bf16(a):
    tmp = a.out / '_tmp'; meta = {}
    dev = a.device
    with torch.inference_mode():
        if 'vision' in a.parts:
            g = VisionGraph(load_vision(device=dev), PPI).eval(); N = PPI * N_IMG
            args = (torch.randn(N, 1536, device=dev), torch.randn(N, 1152, device=dev).to(BF16), torch.randn(N, 72, device=dev), torch.randn(N, 72, device=dev))
            tmp.mkdir(parents=True, exist_ok=True); t0 = time.time()
            torch.onnx.export(g, args, str(tmp / 'vision.onnx'), input_names=['pixel_values', 'pos_embeds', 'cos', 'sin'],
                              output_names=['image_embeds', 'deepstack'], opset_version=19, dynamo=False, external_data=True, do_constant_folding=True)
            meta['vision'] = {'io': _save_onnx(tmp / 'vision.onnx', a.out / 'vision', name='vision'), 'export_s': time.time() - t0}
            shutil.rmtree(tmp); del g; gc.collect(); torch.cuda.empty_cache()
        for i, (lo, hi) in enumerate(bounds()):
            if f'chunk{i}' not in a.parts: continue
            layers, tc = load_text_layers(lo, hi, device=dev); g = TextChunkGraph(layers, tc, lo).eval()
            args = chunk_args(g, device=dev); names = CHUNK_IN + (['deepstack'] if g.has_deepstack else [])
            tmp.mkdir(parents=True, exist_ok=True); t0 = time.time()
            torch.onnx.export(g, args, str(tmp / 'chunk.onnx'), input_names=names, output_names=['h_out', 'new_kv'], opset_version=19,
                              dynamo=False, external_data=True, do_constant_folding=True, dynamic_axes={k: v for k, v in CHUNK_DYN.items() if k in names + ['h_out', 'new_kv']})
            meta[f'chunk{i}'] = {'layers': [lo, hi], 'io': _save_onnx(tmp / 'chunk.onnx', a.out / f'chunk{i}', name='chunk'), 'export_s': time.time() - t0}
            shutil.rmtree(tmp); del g, layers; gc.collect(); torch.cuda.empty_cache()
            print(json.dumps({f'chunk{i}': meta[f'chunk{i}']['export_s']}), flush=True)
        if 'head' in a.parts:
            norm, head = load_head(device=dev); g = HeadGraph(norm, head).eval()
            tmp.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(g, (torch.randn(1, 1, 4096, dtype=BF16, device=dev),), str(tmp / 'head.onnx'), input_names=['h'], output_names=['logits'],
                              opset_version=19, dynamo=False, external_data=True, do_constant_folding=True)
            meta['head'] = {'io': _save_onnx(tmp / 'head.onnx', a.out / 'head', name='head')}
            shutil.rmtree(tmp)
    old = json.loads((a.out / 'export_meta.json').read_text()) if (a.out / 'export_meta.json').exists() else {}
    old.update(meta); old['torch'] = torch.__version__
    (a.out / 'export_meta.json').write_text(json.dumps(old, indent=1)); print('done', list(meta))


# ---------------------------------------------------------------------------------------------------- calibration
def calibdata(a):
    """Roll out the reasoning on the dev clips with the bf16 torch graph; save chunk-0 inputs of prompt + reasoning."""
    import thor_guard; thor_guard.wait_for_free_board(__file__)
    from vlm_runtime import Inputs, Pipeline, TorchBackend
    inp = Inputs(); be = TorchBackend(N_CHUNKS); pipe = Pipeline(be, inp)
    a.out.mkdir(parents=True, exist_ok=True); rows = []
    for p in sorted(CALIB_DIR.glob('*.npz'))[:a.n]:
        x = inp.tokenize(p); r = pipe.run(x)
        n = x['input_ids'].shape[1]; gen = torch.tensor(r['gen'][:-1], device='cuda')   # the fed tokens (last one never fed)
        ids = torch.cat([x['input_ids'][0], gen]); S = ids.shape[0]
        pid = torch.cat([x['position_ids'][:, 0], (torch.arange(n, S, device='cuda') + x['rope_deltas']).view(1, -1).expand(3, -1)], 1)[:, None]
        pos, vcos, vsin = inp.grid_inputs(x['grid'])
        with torch.no_grad():
            img, deep = be.vision(x['pixel_values'].float(), pos, vcos, vsin)
        h = pipe._embeds(ids).view(1, S, -1).clone(); mask = ids == inp.vcfg.image_token_id
        h[0][mask] = img
        d = torch.zeros(3, S, h.shape[-1], dtype=BF16, device='cuda'); d[:, mask] = deep
        cos, sin = inp.text_rope(pid)
        torch.save({'h': h.cpu(), 'deep': d.cpu(), 'cos': cos.cpu(), 'sin': sin.cpu(), 'text': r['text'], 'S': S}, a.out / f'{p.stem}.pt')
        rows.append({'clip': p.stem, 'S': S, 'text': r['text']}); print(json.dumps(rows[-1]), flush=True)
    (a.out / 'calib_index.json').write_text(json.dumps(rows, indent=1))


def _bias(S, device):
    b = torch.full((S, 1 + S), NEG, device=device)
    b[:, 1:] = torch.triu(torch.full((S, S), NEG, device=device), diagonal=1)
    return b


def quant(a):
    import thor_guard; thor_guard.wait_for_free_board(__file__)
    import modelopt.torch.quantization as mtq
    from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata, OnnxBytes
    from modelopt.onnx.export.fp8_exporter import FP8QuantExporter
    FP8QuantExporter._insert_qdq_after_softmax = staticmethod(lambda graph: 0)   # attention stays bf16 (see field)
    import modelopt.torch._deploy.utils.torch_onnx as _tox
    _tox.remove_redundant_casts = lambda graph, *args, **kw: graph               # keep RMSNorm fp32 casts (see field)
    dev = 'cuda'
    files = sorted(a.calib.glob('*.pt'))[:a.n]
    data = [torch.load(f, weights_only=False) for f in files]
    hs = [d['h'] for d in data]                                                   # chunk inputs (bf16 path), CPU
    meta = {'precision': a.precision, 'cfg': a.cfg, 'disable': a.disable, 'calib': [str(f) for f in files], 'chunks': {}}
    a.out.mkdir(parents=True, exist_ok=True)
    for i, (lo, hi) in enumerate(bounds()):
        t0 = time.time()
        layers, tc = load_text_layers(lo, hi, device=dev); g = TextChunkGraph(layers, tc, lo).eval()
        def fwd(model, d, h):
            S = h.shape[1]
            return model(h.to(dev), torch.zeros(hi - lo, 2, 8, 1, 128, dtype=BF16, device=dev), _bias(S, dev),
                         d['cos'].to(dev), d['sin'].to(dev), d['deep'].to(dev) if g.has_deepstack else None)
        with torch.no_grad():
            outs = [fwd(g, d, h)[0].cpu() for d, h in zip(data, hs)]           # bf16 outputs = next chunk's inputs
        if i not in a.chunks or (a.out / f'chunk{i}' / 'chunk.onnx').exists():
            hs = outs; del g, layers; gc.collect(); torch.cuda.empty_cache(); continue
        cfg = copy.deepcopy(getattr(mtq, a.cfg) if a.cfg else (mtq.FP8_DEFAULT_CFG if a.precision == 'fp8' else mtq.NVFP4_DEFAULT_CFG))
        cfg['quant_cfg'] += [{'quantizer_name': p, 'enable': False} for p in a.disable_local(lo, hi)]
        def loop(model):
            with torch.no_grad():
                for d, h in zip(data, hs): fwd(model, d, h)
        g = mtq.quantize(g, cfg, loop)
        with torch.no_grad():
            fq = [float((fwd(g, d, h)[0].float() - o.to(dev).float()).norm() / o.to(dev).float().norm()) for d, h, o in zip(data[:2], hs[:2], outs[:2])]
        nq = sum(1 for n, m in g.named_modules() if n.endswith('_quantizer') and getattr(m, 'is_enabled', False))
        args = chunk_args(g, device=dev); names = CHUNK_IN + (['deepstack'] if g.has_deepstack else [])
        dyn = {k: v for k, v in CHUNK_DYN.items() if k in names}
        b, _ = get_onnx_bytes_and_metadata(g, args, model_name='chunk', dynamic_axes=dyn, weights_dtype='fp32')
        raw = a.out / '_raw'; OnnxBytes.from_bytes(b).write_to_disk(str(raw), clean_dir=True); del b; settle()
        io = _save_onnx(raw / 'chunk.onnx', a.out / f'chunk{i}', out_names=['h_out', 'new_kv'], name='chunk'); shutil.rmtree(raw)
        if a.precision == 'nvfp4': _nvfp4_fix(a.out / f'chunk{i}' / 'chunk.onnx')
        meta['chunks'][f'chunk{i}'] = {'layers': [lo, hi], 'fakequant_rel_l2_vs_bf16': fq, 'enabled_quantizers': nq, 'io': io, 'seconds': time.time() - t0}
        print(json.dumps({f'chunk{i}': {k: v for k, v in meta['chunks'][f'chunk{i}'].items() if k != 'io'}}), flush=True)
        (a.out / f'export_meta_chunk{i}.json').write_text(json.dumps(meta['chunks'][f'chunk{i}'] | {k: meta[k] for k in ('precision', 'cfg', 'disable', 'calib')}, indent=1))
        hs = outs; del g, layers; gc.collect(); torch.cuda.empty_cache()


def _nvfp4_fix(path):
    """deploy/thor/trt/nvfp4_fix.py for one file: fp32 global weight scales -> bf16 (strongly typed bf16 MatMuls)."""
    import numpy as np, onnx
    from onnx import numpy_helper
    m = onnx.load(str(path), load_external_data=False); n = 0
    for i, t in enumerate(m.graph.initializer):
        if t.name.endswith('_f8_scale_f32_scale') and t.data_type == onnx.TensorProto.FLOAT:
            v = numpy_helper.to_array(t).astype(np.float32)
            m.graph.initializer[i].CopyFrom(onnx.helper.make_tensor(t.name, onnx.TensorProto.BFLOAT16, list(t.dims), v.tolist())); n += 1
    onnx.save_model(m, str(path)); print('nvfp4_fix: converted', n, 'global weight scales to bf16', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['bf16', 'calibdata', 'quant']); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parts', nargs='*', default=['vision', 'chunk0', 'chunk1', 'chunk2', 'chunk3', 'head'])
    ap.add_argument('--device', default='cpu'); ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--precision', choices=['fp8', 'nvfp4']); ap.add_argument('--calib', type=Path)
    ap.add_argument('--cfg', default=None, help='ModelOpt config name overriding the precision default')
    ap.add_argument('--disable', nargs='*', default=[], help='global decoder-layer indices kept bf16, e.g. 0 1 34 35')
    ap.add_argument('--chunks', type=int, nargs='*', default=[0, 1, 2, 3])
    a = ap.parse_args()
    # layer indices inside a chunk graph are local: TextChunkGraph.layers.<j>
    a.disable_local = lambda lo, hi: [f'*layers.{int(L) - lo}.*' for L in a.disable if lo <= int(L) < hi]
    {'bf16': export_bf16, 'calibdata': calibdata, 'quant': quant}[a.cmd](a)


if __name__ == '__main__':
    main()
