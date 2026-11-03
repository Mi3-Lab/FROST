"""Build the VLM TensorRT engines from export_vlm.py ONNX (TensorRT Python API; DriveOS 7.0.3 has no trtexec). Thor.

  python build_vlm.py --onnx $FROST_THOR_ROOT/vlm/onnx/bf16 --out $FROST_THOR_ROOT/vlm/engines/bf16 [--parts vision chunk0 ... head]
      [--prefill 3136] [--capacity 3456]

STRONGLY_TYPED networks (dtypes exactly as exported; required for ModelOpt Q/DQ graphs).
  vision : static (shapes of the ONNX: 16 images x 720 patches).
  chunk  : two optimisation profiles sharing the weights:
             profile 0 (prefill): S = --prefill (prompt padded to this bucket; the Alpamayo-1.5 prompt with 16 images
                                  of 196608 px is 3086 tokens), C = --prefill-past masked dummy slots (16: keeps the
                                  key count aligned; with C = 1 attention was 3.8x slower, 17.7 vs 4.7 ms per layer);
             profile 1 (decode) : S = 1, C = --capacity (the KV-cache capacity = the action-expert engine's L).
           --prefill-range MIN MAX makes profile 0 dynamic instead (opt = --prefill).
  head   : static [1,1,4096] -> [1,V].
Writes <out>/<part>.engine, .layers.json (inspector) and .build.json; reuses a timing cache.
"""
import os
import argparse, json, time
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trt'))
import tensorrt as trt  # noqa: E402


def build(onnx_path, out, profiles, a, log):
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, log)
    t0 = time.time()
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors): print('PARSE ERROR', parser.get_error(i))
        raise SystemExit(2)
    parse_s = time.time() - t0
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    cfg.builder_optimization_level = a.opt_level
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache_bytes = a.timing_cache.read_bytes() if a.timing_cache.exists() else b''
    cache = cfg.create_timing_cache(cache_bytes); cfg.set_timing_cache(cache, ignore_mismatch=False)
    shapes = []
    for prof_shapes in profiles:
        prof = builder.create_optimization_profile()
        for i in range(net.num_inputs):
            inp = net.get_input(i); mn, opt, mx = prof_shapes(inp.name, list(inp.shape))
            prof.set_shape(inp.name, mn, opt, mx)
        cfg.add_optimization_profile(prof); shapes.append({net.get_input(i).name: prof_shapes(net.get_input(i).name, list(net.get_input(i).shape)) for i in range(net.num_inputs)})
    t0 = time.time(); plan = builder.build_serialized_network(net, cfg); build_s = time.time() - t0
    if plan is None: raise SystemExit(f'engine build failed: {onnx_path}')
    out.write_bytes(plan); a.timing_cache.write_bytes(memoryview(cfg.get_timing_cache().serialize()))
    import os; os.sync(); time.sleep(5)   # USB SSD: flush every multi-GB engine and pause
    eng = trt.Runtime(log).deserialize_cuda_engine(plan)
    info = eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    Path(str(out) + '.layers.json').write_text(info)
    layers = json.loads(info).get('Layers', [])
    names = [l.get('Name', '') if isinstance(l, dict) else l for l in layers]
    meta = {'trt': trt.__version__, 'onnx': str(onnx_path), 'profiles': shapes, 'opt_level': a.opt_level, 'parse_s': parse_s,
            'build_s': build_s, 'engine_bytes': plan.nbytes, 'n_layers': len(layers),
            'device_memory_per_profile': [eng.get_device_memory_size_for_profile_v2(p) for p in range(eng.num_optimization_profiles)],
            'kernels_sample': [n for n in names if any(s in n.lower() for s in ('mha', 'fmha', 'gemm', 'block_scaled', 'e4m3'))][:12]}
    Path(str(out) + '.build.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ('onnx', 'build_s', 'engine_bytes', 'device_memory_per_profile')}), flush=True)
    del eng, plan, net, parser, builder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--onnx', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parts', nargs='*', default=['vision', 'chunk0', 'chunk1', 'chunk2', 'chunk3', 'head'])
    ap.add_argument('--prefill', type=int, default=3136); ap.add_argument('--prefill-range', type=int, nargs=2, default=None)
    ap.add_argument('--prefill-past', type=int, default=16, help='masked dummy past slots in the prefill profile; '
                    'S + 1 keys (odd) made TRT 10.10 fall back to a 4-6x slower attention kernel, S + 16 keeps the fast one')
    ap.add_argument('--capacity', type=int, default=3456); ap.add_argument('--opt-level', type=int, default=3)
    ap.add_argument('--workspace-gb', type=float, default=4.0)
    ap.add_argument('--timing-cache', type=Path, default=Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/vlm/engines/timing.cache'))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import thor_guard; thor_guard.wait_for_free_board(__file__)
    log = trt.Logger(trt.Logger.WARNING); trt.init_libnvinfer_plugins(log, '')
    Sp = a.prefill; Smin, Smax = a.prefill_range or (Sp, Sp)

    def static(name, shape):
        return shape, shape, shape

    def prefill(name, shape):
        P = a.prefill_past
        def f(S): return {'h': [1, S, 4096], 'past_kv': [shape[0], 2, 8, P, 128], 'bias': [S, P + S],
                          'cos': [S, 128], 'sin': [S, 128], 'deepstack': [3, S, 4096]}[name]
        return f(Smin), f(Sp), f(Smax)

    def decode(name, shape):
        C = a.capacity
        s = {'h': [1, 1, 4096], 'past_kv': [shape[0], 2, 8, C, 128], 'bias': [1, C + 1], 'cos': [1, 128], 'sin': [1, 128], 'deepstack': [3, 1, 4096]}[name]
        return s, s, s

    for part in a.parts:
        out = a.out / f'{part}.engine'
        if out.exists() and not a.force: print(f'{out} exists'); continue
        sub = 'chunk' if part.startswith('chunk') else part
        onnx_path = a.onnx / part / f'{sub}.onnx'
        build(onnx_path, out, [prefill, decode] if part.startswith('chunk') else [static], a, log)


if __name__ == '__main__':
    main()
