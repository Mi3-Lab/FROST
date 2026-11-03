"""Export the Qwen-Drive flow step (expert_graph.ExpertStep) to ONNX with static shapes (workstation CPU, no GPU), with a check against
the PyTorch module on a synthetic cache.  python export_expert.py --model DIR --out DIR [--lmax 4654]"""
import argparse, json, time
from pathlib import Path
import torch
import sys; sys.path.insert(0, str(Path(__file__).parent))
from expert_graph import load_expert, ExpertStep, NEG


def synth_inputs(exp, L, P, seed=0):
    g = torch.Generator().manual_seed(seed); dt = exp.dtype; H = exp.config.hidden_size
    kv = (torch.randn(8, 2, 1, 4, L, 256, generator=g) * torch.tensor([2., 1.]).view(1, 2, 1, 1, 1, 1)).to(dt)
    bias = torch.zeros(L + 50); bias[P:L] = NEG
    anchor = torch.tensor([[P + 20]] * 3); pos = exp._waypoint_positions(anchor, 50)           # anchor [3,1] -> [3,1,50]
    cos, sin = exp.rotary_emb(pos, dt)
    pose = torch.randn(1, H, generator=g).to(dt); vel = torch.randn(1, H, generator=g).to(dt); acc = torch.randn(1, H, generator=g).to(dt)
    cond = torch.randn(1, H, generator=g).to(dt)
    freqs = torch.logspace(0, torch.log10(torch.tensor(16.)).item(), 16, dtype=dt).float()
    return dict(waypoints=torch.randn(1, 50, 3, generator=g), flow_time=torch.tensor([0.3]), freqs=freqs, pose_q=pose, vel_q=vel, acc_q=acc,
                cond_static=cond, cos=cos, sin=sin, past_kv=kv, key_bias=bias), pos


def reference(exp, ins, P, L):
    """PlanningExpert.predict_endpoint with the same externally computed queries/condition (the module computes them inside)."""
    import torch.nn.functional as F
    from expert_graph import rms
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lmax', type=int, default=4654); ap.add_argument('--opset', type=int, default=19); ap.add_argument('--no-export', action='store_true'); ap.add_argument('--dtype', default='bfloat16')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(16)
    exp, cfg = load_expert(a.model, dtype=getattr(torch, a.dtype)); L = a.lmax; P = 4400
    g = ExpertStep(exp, L).eval(); ins, pos = synth_inputs(exp, L, P)
    # check vs the module: scene cache of the real layout [B, L_real, kv, hd] (module concatenates along dim 1), no padding rows
    with torch.inference_mode():
        got = g(**ins)
        cache = [(ins['past_kv'][i, 0].transpose(1, 2)[:, :P], ins['past_kv'][i, 1].transpose(1, 2)[:, :P]) for i in range(8)]
        # drive the module with the same queries by monkeypatching the encoders to return the externally given tensors
        e = exp; nav = torch.tensor([1])
        e.encode_history = lambda *args: (ins['pose_q'], ins['vel_q'], ins['acc_q'])
        orig_nav, orig_ego = e.nav_mlp.forward, e.ego_mlp.forward
        e.nav_mlp.forward = lambda x: ins['cond_static']; e.ego_mlp.forward = lambda x: torch.zeros_like(ins['cond_static'])
        hq = e.encode_history(None)
        ref = e.predict_endpoint(ins['waypoints'], ins['flow_time'], hq, cache, torch.tensor([[P + 20]] * 3), nav, torch.zeros(1, 8))
        e.nav_mlp.forward, e.ego_mlp.forward = orig_nav, orig_ego
    err = {'max_abs': float((got - ref).abs().max()), 'ref_abs_max': float(ref.abs().max()), 'rel_l2': float((got - ref).norm() / ref.norm())}
    print('check', json.dumps(err)); (a.out / 'check.json').write_text(json.dumps(err))
    if a.no_export: return
    t0 = time.time(); names = list(ins)
    with torch.inference_mode():
        torch.onnx.export(g, tuple(ins[k] for k in names), str(a.out / 'step.onnx'), input_names=names, output_names=['endpoint'], opset_version=a.opset,
                          dynamo=False, external_data=True, do_constant_folding=True)
    meta = {'export_s': time.time() - t0, 'lmax': L, 'inputs': {k: [list(v.shape), str(v.dtype)] for k, v in ins.items()}, 'files': {p.name: p.stat().st_size for p in a.out.iterdir()}}
    (a.out / 'export_meta.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta['files'], indent=1))


if __name__ == '__main__':
    main()
