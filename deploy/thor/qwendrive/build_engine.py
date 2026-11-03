"""Static TensorRT engine from an exported Qwen-Drive step ONNX (strongly typed; all shapes fixed in the ONNX). Python API, no trtexec.
python build_engine.py --onnx step.onnx --out step.engine [--opt-level 3]"""
import os
import argparse, json, time
from pathlib import Path
import tensorrt as trt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--onnx', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--opt-level', type=int, default=3); ap.add_argument('--workspace-gb', type=float, default=4.0)
    ap.add_argument('--timing-cache', type=Path, default=Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/qd/engines/timing.cache'))
    a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    log = trt.Logger(trt.Logger.INFO); trt.init_libnvinfer_plugins(log, '')
    b = trt.Builder(log); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)); p = trt.OnnxParser(net, log)
    t0 = time.time()
    if not p.parse_from_file(str(a.onnx)):
        for i in range(p.num_errors): print('PARSE ERROR', p.get_error(i))
        raise SystemExit(2)
    parse_s = time.time() - t0
    cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    cfg.builder_optimization_level = a.opt_level; cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache = cfg.create_timing_cache(a.timing_cache.read_bytes() if a.timing_cache.exists() else b''); cfg.set_timing_cache(cache, ignore_mismatch=False)
    t0 = time.time(); plan = b.build_serialized_network(net, cfg); build_s = time.time() - t0
    if plan is None: raise SystemExit('build failed')
    a.out.write_bytes(plan); a.timing_cache.parent.mkdir(parents=True, exist_ok=True); a.timing_cache.write_bytes(memoryview(cfg.get_timing_cache().serialize()))
    eng = trt.Runtime(log).deserialize_cuda_engine(plan)
    meta = {'trt': trt.__version__, 'onnx': str(a.onnx), 'parse_s': parse_s, 'build_s': build_s, 'engine_bytes': plan.nbytes,
            'device_memory_bytes': eng.device_memory_size_v2 if hasattr(eng, 'device_memory_size_v2') else eng.device_memory_size,
            'io': [(eng.get_tensor_name(i), str(eng.get_tensor_dtype(eng.get_tensor_name(i))), list(eng.get_tensor_shape(eng.get_tensor_name(i)))) for i in range(eng.num_io_tensors)]}
    Path(str(a.out) + '.build.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
