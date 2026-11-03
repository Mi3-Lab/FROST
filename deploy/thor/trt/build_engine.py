"""Build a static TensorRT engine for the velocity field from field.onnx (TensorRT Python API; no trtexec on DriveOS).

  python build_engine.py --onnx $FROST_THOR_ROOT/trt/onnx/onnx_fused/field.onnx --batch 1 --lmax 4096 \
      --out $FROST_THOR_ROOT/trt/engines/fused_b1_L4096.engine [--opt-level 3] [--profiles 1 6]

The network is STRONGLY_TYPED: every tensor keeps the dtype of the ONNX graph (bf16 GEMMs, fp32 residual stream,
RMSNorm and softmax as exported), which is also what explicitly quantized (ModelOpt Q/DQ) graphs require. The ONNX
has dynamic B and L; each optimization profile pins them (min = opt = max), so every engine is static. With
--profiles 1 6 one engine holds two profiles (B=1 and B=6) that share the weights.
Writes <out>, <out>.layers.json (engine inspector, DETAILED) and <out>.build.json; reuses a timing cache.
"""
import argparse, json, os, time
import thor_guard; thor_guard.wait_for_free_board(__file__)   # before any CUDA use (see thor_guard.py)
from pathlib import Path
import tensorrt as trt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--onnx', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=1); ap.add_argument('--profiles', type=int, nargs='*', default=None)
    ap.add_argument('--lmax', type=int, default=4096); ap.add_argument('--opt-level', type=int, default=3)
    ap.add_argument('--workspace-gb', type=float, default=4.0)
    ap.add_argument('--timing-cache', type=Path, default=Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/trt/engines/timing.cache'))
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    log = trt.Logger(trt.Logger.VERBOSE if a.verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(log, '')
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, log)
    t0 = time.time()
    if not parser.parse_from_file(str(a.onnx)):
        for i in range(parser.num_errors): print('PARSE ERROR', parser.get_error(i))
        raise SystemExit(2)
    parse_s = time.time() - t0
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    cfg.builder_optimization_level = a.opt_level
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache_bytes = a.timing_cache.read_bytes() if a.timing_cache.exists() else b''
    cache = cfg.create_timing_cache(cache_bytes); cfg.set_timing_cache(cache, ignore_mismatch=False)
    batches = a.profiles or [a.batch]
    for B in batches:
        prof = builder.create_optimization_profile()
        for i in range(net.num_inputs):
            inp = net.get_input(i)
            shape = [B if d == -1 and inp.name in ('x', 't') else d for d in inp.shape]
            shape = [a.lmax if d == -1 else d for d in shape]
            prof.set_shape(inp.name, shape, shape, shape)
        cfg.add_optimization_profile(prof)
    t0 = time.time(); plan = builder.build_serialized_network(net, cfg); build_s = time.time() - t0
    if plan is None: raise SystemExit('engine build failed')
    a.out.write_bytes(plan)
    a.timing_cache.write_bytes(memoryview(cfg.get_timing_cache().serialize()))
    eng = trt.Runtime(log).deserialize_cuda_engine(plan)
    info = eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    Path(str(a.out) + '.layers.json').write_text(info)
    layers = json.loads(info).get('Layers', [])
    names = [l.get('Name', '') if isinstance(l, dict) else l for l in layers]
    meta = {'trt': trt.__version__, 'onnx': str(a.onnx), 'batches': batches, 'lmax': a.lmax, 'opt_level': a.opt_level,
            'parse_s': parse_s, 'build_s': build_s, 'engine_bytes': plan.nbytes, 'n_layers': len(layers),
            'device_memory_bytes': eng.device_memory_size_v2 if hasattr(eng, 'device_memory_size_v2') else eng.device_memory_size,
            'io': [(eng.get_tensor_name(i), str(eng.get_tensor_dtype(eng.get_tensor_name(i))),
                    list(eng.get_tensor_shape(eng.get_tensor_name(i)))) for i in range(eng.num_io_tensors)],
            'mha_like_layers': [n for n in names if any(s in n.lower() for s in ('mha', 'fmha', 'attention', '_myl'))][:20]}
    Path(str(a.out) + '.build.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
