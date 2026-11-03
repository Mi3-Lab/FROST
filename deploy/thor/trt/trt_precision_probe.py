"""Does TensorRT 10.10 on DRIVE Thor (sm_101) build and run bf16 / FP8 / NVFP4 GEMMs, and how fast?

Single-GEMM networks with the expert's shapes, Y[M,N] = X[M,K] @ W[N,K]^T, X a bf16 network input:
  bf16  : W bf16 constant.
  fp8   : X -> Quantize(per-tensor scale, FP8) -> Dequantize; W bf16 constant -> Quantize(per-tensor) -> Dequantize
          (the explicit-quantization pattern of a ModelOpt FP8 ONNX; TRT folds the weight Q into FP8 weights).
  nvfp4 : X -> DynamicQuantize(axis -1, block 16, FP4 data + FP8 block scales, fp32 global double-quant scale)
          -> Dequantize(scales) -> Dequantize(data); W given as FP4 constant (packed e2m1) with FP8 block scales and
          an fp32 global scale (the ModelOpt NVFP4 weight layout). Both nibble orders are tried for the packing.
Numerics: TRT output vs (a) the exact bf16 GEMM, (b) a torch emulation of the same quantization.
Timing: CUDA events, 50 warm-up + 300 runs, median us and effective weight bandwidth.

  python trt_precision_probe.py --out $FROST_THOR_ROOT/trt/out/precision_probe.json
"""
import argparse, json, time, traceback
import thor_guard; thor_guard.wait_for_free_board(__file__)   # before any CUDA use (see thor_guard.py)
import numpy as np
import torch
import tensorrt as trt

LOG = trt.Logger(trt.Logger.WARNING)
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_keep = []   # host buffers referenced by trt.Weights must outlive the build


def W(t):
    t = t.contiguous().cpu(); _keep.append(t)
    dt = {torch.bfloat16: trt.bfloat16, torch.float32: trt.float32, torch.float16: trt.float16}[t.dtype]
    return trt.Weights(dt, t.data_ptr(), t.numel())


def Wraw(dtype, u8, count):
    u8 = u8.contiguous().cpu(); _keep.append(u8); return trt.Weights(dtype, u8.data_ptr(), count)


def fp8_bytes(x):
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def quant_fp4_blocks(x, gscale):
    """x [R, K] fp32 -> (codes [R,K] int 0..15, block scales fp8-rounded fp32 [R, K/16]); NVFP4 with global scale."""
    R, K = x.shape; xb = x.view(R, K // 16, 16)
    s = (xb.abs().amax(-1) / 6.0 / gscale).clamp(min=2 ** -9)
    s = s.to(torch.float8_e4m3fn).float()                     # fp8 block scale (relative to the global scale)
    y = xb / (s * gscale)[..., None]
    mag = (y.abs()[..., None] - E2M1).abs().argmin(-1)        # nearest e2m1 magnitude
    codes = mag + 8 * (y < 0).long()
    return codes.view(R, K), s


def dequant_fp4(codes, s, gscale):
    R, K = codes.shape
    v = E2M1[codes % 8] * torch.where(codes >= 8, -1., 1.)
    return (v.view(R, K // 16, 16) * (s * gscale)[..., None]).view(R, K)


def pack_fp4(codes, low_first=True):
    c = codes.flatten().to(torch.uint8); a, b = c[0::2], c[1::2]
    return (a | (b << 4)) if low_first else ((a << 4) | b)


def build(make, name):
    b = trt.Builder(LOG); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    make(net); cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    t = time.time(); plan = b.build_serialized_network(net, cfg)
    if plan is None: raise RuntimeError(f'{name}: build_serialized_network returned None')
    eng = trt.Runtime(LOG).deserialize_cuda_engine(plan)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    return eng, time.time() - t, [l if isinstance(l, str) else {k: l.get(k) for k in ('Name', 'LayerType', 'TacticName', 'Inputs', 'Outputs') if k in l} for l in info.get('Layers', [])]


def run(eng, X, iters=300):
    ctx = eng.create_execution_context(); s = torch.cuda.Stream()
    Y = torch.empty(tuple(eng.get_tensor_shape('Y')), dtype=torch.bfloat16, device='cuda')
    ctx.set_tensor_address('X', X.data_ptr()); ctx.set_tensor_address('Y', Y.data_ptr())
    with torch.cuda.stream(s):
        for _ in range(50): ctx.execute_async_v3(s.cuda_stream)
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for a, b in ev: a.record(s); ctx.execute_async_v3(s.cuda_stream); b.record(s)
    s.synchronize()
    return Y.float(), float(np.median([a.elapsed_time(b) * 1e3 for a, b in ev]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--M', type=int, nargs='+', default=[64, 384])
    ap.add_argument('--shapes', default='2048x8256,8256x2048,2048x2048', help='KxN list')
    a = ap.parse_args(); torch.manual_seed(0)
    res = {'trt': trt.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
           'capability': torch.cuda.get_device_capability(0), 'cases': []}
    for shp in a.shapes.split(','):
        K, N = map(int, shp.split('x'))
        Wt = (torch.randn(N, K) * K ** -0.5)
        for M in a.M:
            X = torch.randn(M, K, device='cuda').to(torch.bfloat16)
            ref = (X.float() @ Wt.to(torch.bfloat16).float().cuda().T)
            base = {'M': M, 'K': K, 'N': N}

            def gemm_in(net):
                return net.add_input('X', trt.bfloat16, (M, K))

            def out(net, y): y.name = 'Y'; net.mark_output(y)

            def mk_bf16(net):
                x = gemm_in(net); w = net.add_constant((N, K), W(Wt.to(torch.bfloat16))).get_output(0)
                out(net, net.add_matrix_multiply(x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE).get_output(0))

            sx = float(X.float().abs().max() / 448); sw = float(Wt.abs().max() / 448)

            def mk_fp8(net):
                x = gemm_in(net)
                csx = net.add_constant((), W(torch.tensor(sx, dtype=torch.bfloat16))).get_output(0)
                csw = net.add_constant((), W(torch.tensor(sw, dtype=torch.bfloat16))).get_output(0)
                xq = net.add_quantize(x, csx, trt.fp8).get_output(0); xd = net.add_dequantize(xq, csx, trt.bfloat16).get_output(0)
                w = net.add_constant((N, K), W(Wt.to(torch.bfloat16))).get_output(0)
                wq = net.add_quantize(w, csw, trt.fp8).get_output(0); wd = net.add_dequantize(wq, csw, trt.bfloat16).get_output(0)
                out(net, net.add_matrix_multiply(xd, trt.MatrixOperation.NONE, wd, trt.MatrixOperation.TRANSPOSE).get_output(0))

            def emu_fp8():
                xq = (X.float() / sx).to(torch.float8_e4m3fn).float() * sx
                wq = (Wt.cuda() / sw).to(torch.float8_e4m3fn).float() * sw
                return xq @ wq.T

            gw = float(Wt.abs().max() / (6 * 448)); gx = float(X.float().abs().max() / (6 * 448))
            codes, sblk = quant_fp4_blocks(Wt.float(), gw)

            def mk_fp4(low_first):
                def mk(net):
                    x = gemm_in(net)
                    gxs = net.add_constant((), W(torch.tensor(gx, dtype=torch.float32))).get_output(0)
                    dq = net.add_dynamic_quantize(x, 1, 16, trt.fp4, trt.fp8); dq.set_input(1, gxs)
                    xs = net.add_dequantize(dq.get_output(1), gxs, trt.bfloat16).get_output(0)   # FP8 block scales -> bf16
                    xd = net.add_dequantize(dq.get_output(0), xs, trt.bfloat16).get_output(0)
                    w4 = net.add_constant((N, K), Wraw(trt.fp4, pack_fp4(codes, low_first), N * K)).get_output(0)
                    ws8 = net.add_constant((N, K // 16), Wraw(trt.fp8, fp8_bytes(sblk), N * K // 16)).get_output(0)
                    gws = net.add_constant((), W(torch.tensor(gw, dtype=torch.float32))).get_output(0)
                    wsc = net.add_dequantize(ws8, gws, trt.bfloat16).get_output(0)
                    wd = net.add_dequantize(w4, wsc, trt.bfloat16).get_output(0)
                    out(net, net.add_matrix_multiply(xd, trt.MatrixOperation.NONE, wd, trt.MatrixOperation.TRANSPOSE).get_output(0))
                return mk

            def emu_fp4():
                xc, xs = quant_fp4_blocks(X.float().cpu(), gx)
                return (dequant_fp4(xc, xs, gx) @ dequant_fp4(codes, sblk, gw).T).cuda()

            for prec, mk, emu, wbytes in [('bf16', mk_bf16, None, 2 * N * K), ('fp8', mk_fp8, emu_fp8, N * K),
                                          ('nvfp4_lowfirst', mk_fp4(True), emu_fp4, N * K // 2 + N * K // 16),
                                          ('nvfp4_highfirst', mk_fp4(False), emu_fp4, N * K // 2 + N * K // 16)]:
                case = dict(base, precision=prec)
                try:
                    eng, bt, layers = build(mk, prec); Y, us = run(eng, X)
                    case.update(ok=True, build_s=bt, median_us=us, weight_GBps=wbytes / us / 1e3,
                                rel_err_vs_bf16=float((Y - ref).norm() / ref.norm()), layers=layers)
                    if emu is not None:
                        e = emu(); case['rel_err_vs_emulation'] = float((Y - e).norm() / e.norm())
                        case['emulation_rel_err_vs_bf16'] = float((e - ref).norm() / ref.norm())
                except Exception as ex:
                    case.update(ok=False, error=f'{type(ex).__name__}: {ex}', tb=traceback.format_exc()[-800:])
                res['cases'].append(case)
                print(json.dumps({k: v for k, v in case.items() if k not in ('layers', 'tb')}), flush=True)
                _keep.clear(); torch.cuda.empty_cache()
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
