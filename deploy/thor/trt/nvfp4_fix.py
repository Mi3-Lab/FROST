"""Make a ModelOpt NVFP4 ONNX of the bf16 field parse under TensorRT's strongly typed mode.
ModelOpt (0.47, weights_dtype='fp32') dequantizes FP4 weights with an fp32 per-tensor scale, so the weight branch of every
MatMul is fp32 while the activation branch is bf16 (TRT: 'IMatrixMultiplyLayer must have same input types').
Fix: store each '*_f8_scale_f32_scale' (per-tensor global weight scale) as bf16; the two weight DequantizeLinear nodes then
output bf16 (DQ output type = scale type). Rewrites only the graph file (external weight data untouched).
  python nvfp4_fix.py DIR"""
import sys
import numpy as np, onnx
from onnx import numpy_helper
d = sys.argv[1]
m = onnx.load(f'{d}/field.onnx', load_external_data=False); n = 0
for i, t in enumerate(m.graph.initializer):
    if t.name.endswith('_f8_scale_f32_scale') and t.data_type == onnx.TensorProto.FLOAT:
        v = numpy_helper.to_array(t).astype(np.float32)
        new = onnx.helper.make_tensor(t.name, onnx.TensorProto.BFLOAT16, list(t.dims), v.tolist())
        m.graph.initializer[i].CopyFrom(new); n += 1
onnx.save_model(m, f'{d}/field.onnx'); print('converted', n, 'global weight scales to bf16')
