"""TensorRT backend of vlm_runtime.Pipeline: runs the engines built by build_vlm.py.

ENGINE_DIR layout: vision.engine, chunk0..3.engine (profile 0 = prefill, profile 1 = decode), head.engine.
All engines share one scratch (activation) buffer: they never run concurrently (one stream), so the device memory
is max(engine, profile) instead of the sum. Engines may come from different precision directories (e.g. bf16 vision
and head with FP8 chunks): pass --vision/--head overrides, or put symlinks in the directory.
"""
import json
from pathlib import Path

import torch
import tensorrt as trt

TDT = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16, trt.float16: torch.float16, trt.int32: torch.int32,
       trt.int64: torch.int64}
LOG = trt.Logger(trt.Logger.WARNING)


class TorchGpuAllocator(trt.IGpuAsyncAllocator):
    """TensorRT device allocations through PyTorch's caching allocator. On DriveOS 7.0.3 TensorRT's default
    (cudaMallocAsync) allocator failed at ~8 GB, while the cudaMalloc allocations PyTorch makes from the 26 GiB
    hugetlbfs pool succeed; routing engine weights and scratch through torch makes them share that pool."""

    def __init__(self):
        trt.IGpuAsyncAllocator.__init__(self); self.live = {}

    def _alloc(self, size, alignment):
        t = torch.empty(int(size) + int(alignment), dtype=torch.uint8, device='cuda')
        p = t.data_ptr(); p += (-p) % max(int(alignment), 1)
        self.live[p] = t; return p

    def allocate_async(self, size, alignment, flags, stream):
        return self._alloc(size, alignment)

    def deallocate_async(self, memory, stream):
        return self.live.pop(int(memory), None) is not None

    def allocate(self, size, alignment, flags):
        return self._alloc(size, alignment)

    def deallocate(self, memory):
        return self.live.pop(int(memory), None) is not None


ALLOCATOR = None


def runtime():
    global ALLOCATOR
    trt.init_libnvinfer_plugins(LOG, '')
    rt = trt.Runtime(LOG)
    if ALLOCATOR is None: ALLOCATOR = TorchGpuAllocator()
    rt.gpu_allocator = ALLOCATOR
    return rt


class Engine:
    def __init__(self, path, runtime):
        self.path = Path(path)
        self.eng = runtime.deserialize_cuda_engine(self.path.read_bytes())
        assert self.eng is not None, path
        self.n_prof = self.eng.num_optimization_profiles
        self.ctx = []
        for p in range(self.n_prof):
            c = self.eng.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            c.set_optimization_profile_async(p, torch.cuda.current_stream().cuda_stream)
            self.ctx.append(c)
        torch.cuda.synchronize()
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inputs = [n for n in self.names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.dtype = {n: TDT[self.eng.get_tensor_dtype(n)] for n in self.names}
        self._sinks = {}
        self.outputs = [n for n in self.outputs if not n.startswith('sink_')] + [n for n in self.outputs if n.startswith('sink_')]

    def scratch_bytes(self):
        return max(self.eng.get_device_memory_size_for_profile_v2(p) for p in range(self.n_prof))

    def bind_scratch(self, buf):
        for c in self.ctx: c.set_device_memory(buf.data_ptr(), buf.numel())

    def profile_shape(self, name, p):
        return [tuple(s) for s in self.eng.get_tensor_profile_shape(name, p)]

    def __call__(self, p, feeds, outs=None):
        """feeds: {input name: tensor}; outs: optional preallocated {output name: tensor}. Returns outputs."""
        c = self.ctx[p]
        for n in self.inputs:
            t = feeds[n]
            assert t.dtype == self.dtype[n] and t.is_contiguous(), (self.path.name, n, t.dtype, self.dtype[n])
            c.set_input_shape(n, tuple(t.shape)); c.set_tensor_address(n, t.data_ptr())
        outs = dict(outs or {})
        for n in self.outputs:
            if n.startswith('sink_'):      # unused outputs that only keep the attention output in bf16 (attn_surgery.py): one shared buffer
                shp = tuple(c.get_tensor_shape(n)); key = (shp, self.dtype[n])
                if key not in self._sinks: self._sinks[key] = torch.empty(shp, dtype=self.dtype[n], device='cuda')
                c.set_tensor_address(n, self._sinks[key].data_ptr()); continue
            if n not in outs: outs[n] = torch.empty(tuple(c.get_tensor_shape(n)), dtype=self.dtype[n], device='cuda')
            c.set_tensor_address(n, outs[n].data_ptr())
        assert c.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return outs


class TRTBackend:
    def __init__(self, engine_dir, vision=None, head=None, n_chunks=4, ppi=720, decode_dir=None):
        self.rt = runtime()
        d = Path(engine_dir); self.ppi = ppi
        self.vis = Engine(vision or d / 'vision.engine', self.rt)
        self.chunks = [Engine(d / f'chunk{i}.engine', self.rt) for i in range(n_chunks)]
        self.head_e = Engine(head or d / 'head.engine', self.rt)
        # optional different precision for the decode profile (e.g. FP8 prefill + NVFP4 decode): chunks of decode_dir, profile 1
        self.chunks_dec = [Engine(Path(decode_dir) / f'chunk{i}.engine', self.rt) for i in range(n_chunks)] if decode_dir else self.chunks
        self.bounds = [(i * 36 // n_chunks, (i + 1) * 36 // n_chunks) for i in range(n_chunks)]
        engines = [self.vis, self.head_e] + self.chunks + ([e for e in self.chunks_dec if e not in self.chunks])
        self.scratch = torch.empty(max(e.scratch_bytes() for e in engines), dtype=torch.uint8, device='cuda')
        for e in engines: e.bind_scratch(self.scratch)
        mn, _, mx = self.chunks[0].profile_shape('h', 0)
        self.prefill_bucket = mn[1] if mn[1] == mx[1] else None   # static prefill profile: pad the prompt to it
        self.prefill_past = self.chunks[0].profile_shape('past_kv', 0)[0][3]
        self.meta = {e.path.name: json.loads(Path(str(e.path) + '.build.json').read_text()) if Path(str(e.path) + '.build.json').exists() else None for e in engines}

    def vision(self, pix, pos, cos, sin):
        o = self.vis(0, {'pixel_values': pix.contiguous(), 'pos_embeds': pos.contiguous(), 'cos': cos.contiguous(), 'sin': sin.contiguous()})
        return o['image_embeds'], o['deepstack']

    def chunk(self, i, h, past, bias, cos, sin, deep, mode):
        e = self.chunks[i] if mode == 'prefill' else self.chunks_dec[i]; p = 0 if mode == 'prefill' else 1
        feeds = {'h': h.contiguous(), 'past_kv': past, 'bias': bias.contiguous(), 'cos': cos.contiguous(), 'sin': sin.contiguous()}
        if 'deepstack' in e.inputs: feeds['deepstack'] = deep.contiguous()
        o = e(p, feeds)
        return o['h_out'], o['new_kv']

    def head(self, h):
        return self.head_e(0, {'h': h.contiguous()})['logits']


class FieldEngine:
    """The action-expert velocity-field engine of deploy/thor/trt (x, t, past_kv, key_bias, cos, sin -> v) on the same
    allocator. Callable as f(x, t) for frost.probes.temperature_probe.integrate; past_kv is bound to the VLM cache tensor
    itself (zero-copy handoff)."""

    def __init__(self, path, rt=None):
        self.e = Engine(path, rt or runtime())
        self.scratch = torch.empty(self.e.scratch_bytes(), dtype=torch.uint8, device='cuda'); self.e.bind_scratch(self.scratch)
        shp = {n: self.e.profile_shape(n, 0)[2] for n in self.e.inputs}
        self.buf = {n: torch.zeros(shp[n], dtype=self.e.dtype[n], device='cuda') for n in self.e.inputs if n != 'past_kv'}
        self.B, self.L = shp['x'][0], shp['key_bias'][0]
        self.kv = None; self.out = None

    def bind(self, kv, key_bias, cos, sin):
        assert kv.shape[3] == self.L and kv.is_contiguous(), (kv.shape, self.L)
        self.kv = kv; self.buf['key_bias'].copy_(key_bias); self.buf['cos'].copy_(cos); self.buf['sin'].copy_(sin)

    def __call__(self, x, t):
        self.buf['x'].copy_(x); self.buf['t'].copy_(t.reshape(self.buf['t'].shape))
        o = self.e(0, {**self.buf, 'past_kv': self.kv}, self.out)
        self.out = o
        return o['v'].clone()
