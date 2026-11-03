"""Graph surgery on an exported FP8 chunk ONNX (graph only; the weights file is symlinked, nothing big is written). Thor.

  python attn_surgery.py --src ../../onnx/fp8_v2 --dst ../../onnx/fp8_x1 --variant castpair --parts chunk1

Why: in the FP8 chunk the o_proj input quantizer is fused into the fused-MHA kernel (FP8 output) and that tactic is 2.6x
slower (7.6 ms/layer) than the bf16-output one (2.9 ms/layer), see profile_engine.py. Variants of the node chain
Reshape_6 -> o_proj/input_quantizer/QuantizeLinear:
  castpair  bf16 -> fp32 -> bf16 casts between the attention output and the quantizer (value exact)
  sideout   the attention output is also a (unused) graph output sink_<k>, so it must be materialised in bf16 (the TRT runtime
            Engine binds all sink_* outputs to one shared scratch tensor; measured: chunk prefill 124.6 -> 85.6 ms)
  ident     an Identity node (control; usually removed by the parser)
"""
import argparse, os, re, shutil
from pathlib import Path
import onnx
from onnx import helper, TensorProto


def apply(m, variant):
    g = m.graph; nodes = list(g.node); new = []; n_done = 0
    for n in nodes:
        if n.op_type == 'QuantizeLinear' and re.fullmatch(r'/o_proj/input_quantizer(_\d+)?/TRT_FP8QuantizeLinear', n.name):
            src = n.input[0]; tag = n.name
            if variant == 'castpair':
                a, b = f'{src}_f32_{n_done}', f'{src}_bf16b_{n_done}'
                new.append(helper.make_node('Cast', [src], [a], name=f'cp_a_{n_done}', to=TensorProto.FLOAT))
                new.append(helper.make_node('Cast', [a], [b], name=f'cp_b_{n_done}', to=TensorProto.BFLOAT16))
                n.input[0] = b
            elif variant == 'ident':
                b = f'{src}_id_{n_done}'; new.append(helper.make_node('Identity', [src], [b], name=f'id_{n_done}')); n.input[0] = b
            elif variant == 'sideout':      # the tensor is renamed sink_<k>; vlm_trt.Engine binds every sink_* output to one shared buffer
                sink = f'sink_{n_done}'
                for m2 in nodes:
                    for j, o in enumerate(m2.output):
                        if o == src: m2.output[j] = sink
                n.input[0] = sink
                g.output.append(helper.make_tensor_value_info(sink, TensorProto.BFLOAT16, None))
            else: raise SystemExit(variant)
            n_done += 1
        new.append(n)
    del g.node[:]; g.node.extend(new)
    return n_done


def apply_fp8mha(m, scales=None):
    """FP8 Q/DQ on both attention matmuls (Q, K^T, softmax output, V): the pattern TensorRT fuses into an FP8 MHA kernel.
    scales: None -> reuse the o_proj activation scale of the same graph (timing probe only), or a dict name -> float."""
    g = m.graph; prod = {o: n for n in g.node for o in n.output}; cons = {}
    for n in g.node:
        for i in n.input: cons.setdefault(i, []).append(n)
    inits = {i.name for i in g.initializer}
    scale_src = [i.name for i in g.initializer if i.name.endswith('input_quantizer/Constant_output_0') or 'input_quantizer/Constant_output_0' in i.name]
    zp = [i.name for i in g.initializer if i.name.endswith('TRT_FP8QuantizeLinear_zero_point')][0]
    sc = scale_src[0] if scale_src else None
    assert sc, 'no activation scale initializer found'
    new_nodes = {}; k = 0
    def qdq(x, tag):
        nonlocal k
        q, d = f'att_q_{k}', f'att_dq_{k}'
        a = helper.make_node('QuantizeLinear', [x, sc, zp], [q], name=f'att_Q_{k}', domain='trt', saturate=1)
        b = helper.make_node('DequantizeLinear', [q, sc], [d], name=f'att_DQ_{k}', domain='trt')
        k += 1; return [a, b], d
    for sm in [n for n in g.node if n.op_type == 'Softmax']:
        add = prod[sm.input[0]]; mul = prod[add.input[0]]; mm1 = prod[mul.input[0]]; mm2 = cons[sm.output[0]][0]
        assert mm1.op_type == 'MatMul' and mm2.op_type == 'MatMul'
        ins = []
        for node, idx in ((mm1, 0), (mm1, 1), (mm2, 0), (mm2, 1)):
            nodes_, d = qdq(node.input[idx], idx); ins += nodes_; node.input[idx] = d
        new_nodes[mm1.name] = ins[:4]; new_nodes[mm2.name] = ins[4:]
    out = []
    for n in g.node:
        if n.name in new_nodes: out += new_nodes[n.name]
        out.append(n)
    del g.node[:]; g.node.extend(out); return k // 4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, required=True); ap.add_argument('--dst', type=Path, required=True)
    ap.add_argument('--variant', required=True); ap.add_argument('--parts', nargs='+', default=['chunk0', 'chunk1', 'chunk2', 'chunk3'])
    a = ap.parse_args()
    for part in a.parts:
        d = a.dst / part; d.mkdir(parents=True, exist_ok=True)
        m = onnx.load(str(a.src / part / 'chunk.onnx'), load_external_data=False)
        n = apply_fp8mha(m) if a.variant == 'fp8mha' else apply(m, a.variant)
        onnx.save(m, str(d / 'chunk.onnx'))     # external-data references are kept (same relative file name)
        link = d / 'chunk.onnx.data'
        if not link.exists(): os.symlink((a.src / part / 'chunk.onnx.data').resolve(), link)
        print(part, 'modified', n, 'attention outputs', flush=True)


if __name__ == '__main__':
    main()
