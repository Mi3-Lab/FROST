"""Re-save an exported ONNX model with all weights in one external-data file and print an I/O + op summary.
  python onnx_pack.py SRC_DIR DST_DIR"""
import collections, json, sys
from pathlib import Path
import onnx
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
m = onnx.load(str(src / 'field.onnx'), load_external_data=True)
io = {'inputs': [(i.name, i.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
      'outputs': [(i.name, i.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.output],
      'opset': [(o.domain, o.version) for o in m.opset_import],
      'ops': dict(collections.Counter(n.op_type for n in m.graph.node).most_common()),
      'initializer_dtypes': dict(collections.Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer))}
onnx.save_model(m, str(dst / 'field.onnx'), save_as_external_data=True, all_tensors_to_one_file=True, location='field.onnx.data', size_threshold=1024)
for f in src.glob('*.json'): (dst / f.name).write_bytes(f.read_bytes())
(dst / 'graph_summary.json').write_text(json.dumps(io, indent=1)); print(json.dumps(io, indent=1))
