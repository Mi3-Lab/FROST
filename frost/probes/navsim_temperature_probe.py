"""Numerical-temperature probe on a second family of planners (NAVSIM navtest).

Pre-registered in protocols/navsim.md; setup in README.md (NAVSIM).
Nothing in the upstream repositories is edited: the upstream sampler is wrapped at runtime from this file. For every scene the
VLM / backbone runs once, then for every seed the initial noise is drawn once and every config is integrated from it.

Run (ReCogDrive needs its Python 3.10 environment, envs/navsim_recogdrive.txt):
  source configs/navsim.env; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  PYTHONPATH=.:$RECOGDRIVE_SRC python -m frost.probes.navsim_temperature_probe --model recogdrive_rl --out ...
  PYTHONPATH=.:$DIFFUSIONDRIVE_SRC python -m frost.probes.navsim_temperature_probe --model diffusiondrive --out ...

Plan keys: {token}_s{seed}_{config} and {token}_human; each value is (8,3) (x,y,heading) at 0.5 s in the ego frame.
Config names (Addendum 1): ReCogDrive ddim{K}_trail_eta0_t{tau}, ddim5_lead_eta0_t1.0, release5_lead_eta1_t1.0; DiffusionDrive ddim9_int_eta0_t{tau}, ddim5_int_eta0_t1.0, release2_lead_eta0_t1.0.

Hooks (upstream file:line, commit pins in THIRD_PARTY.md)
ReCogDrive ($RECOGDRIVE_SRC, navsim/agents/recogdrive):
  H1 recogdrive_agent.py:209  ReCogDriveAgent.forward calls action_head.get_action(last_hidden_state, action_inputs) after the
     VLM forward (L189 outputs.hidden_states[-1]). We replace the instance attribute action_head.get_action: the VLM has
     already run once; our wrapper integrates all configs from that hidden state and returns the release plan.
  H2 recogdrive_diffusion_planner.py:539-549 (get_action prologue: feature_encoder, his_traj_encoder, ego_status_encoder)
     reproduced verbatim in `embed`.
  H3 recogdrive_diffusion_planner.py:550-552 initial noise randn((B,8,3)): replaced by tau*z with z from a per-seed
     torch.Generator (CPU, then moved), so it is drawn once per seed and shared by all configs.
  H4 recogdrive_diffusion_planner.py:612-633 DDIM loop reproduced in `ddim_loop`; each step calls upstream
     p_mean_variance (L379-443) unchanged, with deterministic=True (eta=0: sigma and per-step noise zeroed as in L627)
     or False (EtaFixed eta=1, noise clipped to eval_randn_clip_value, std floor eval_min_sampling_denoising_std, L618-629).
  H5 recogdrive_diffusion_planner.py:223-258 _init_ddim_sampler: the DDIM time grid is arange(0,K)*(100//K) flipped
     (K=5 -> 80,60,40,20,0). Addendum 1: the eta=0 sweep instead uses trailing spacing t_i=round(100-i*100/K)-1 with the same
     formulas for alphas/alphas_prev; the leading K=5 buffers come from upstream init at load time and the release buffers are restored; p_mean_variance reads the swapped buffers.
  H6 recogdrive_diffusion_planner.py:635-637 final clamp to final_action_clip_value and denorm_odo (L897).
DiffusionDrive v1 ($DIFFUSIONDRIVE_SRC, navsim/agents/diffusiondrive/transfuser_model_v2.py):
  H7 L504-558 TrajectoryHead.forward_test is called from V2TransfuserModel.forward (L132 via L456). We replace the instance
     attribute _trajectory_head.forward_test; after the backbone ran once we reproduce L505-558 in `dd_loop`.
  H8 L505 step_num (2 in the release) -> parameter; L509-511 roll timesteps (arange(step_num)*20/step_num).round()[::-1].
  H9 L517-519 initial noise: torch.randn(img.shape) -> tau*z (z per-seed generator), add_noise at t=8 (truncation).
  H10 L545-552 diffusion_scheduler.step (DDIMScheduler eta=0, prediction_type=sample) unchanged; L555-558 mode by argmax poses_cls.
CPU-only smoke mode (no CUDA): upstream hard-codes .cuda() / torch.device('cuda') (recogdrive_agent.py:93,129,133,142,191;
recogdrive_backbone.py:118; recogdrive_features.py). `cpu_compat` neutralizes them at runtime and casts the fp16 head to fp32.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

os.environ.setdefault('OMEGACONF_MAX_YAML_EXPANDED_NODES', 'none')  # trusted navtest.yaml with a long token list
ARTIFACTS = Path(os.environ.get('FROST_ARTIFACTS', 'artifacts'))
CKPT = Path(os.environ.get('NAVSIM_CKPT', 'checkpoints/navsim'))
REPOS = {'recogdrive_rl': Path(os.environ.get('RECOGDRIVE_SRC', 'third_party/recogdrive')), 'recogdrive_il': Path(os.environ.get('RECOGDRIVE_SRC', 'third_party/recogdrive')), 'diffusiondrive': Path(os.environ.get('DIFFUSIONDRIVE_SRC', 'third_party/diffusiondrive'))}
CHECKPOINTS = {
    'recogdrive_rl': {'planner': CKPT / 'ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt', 'vlm': CKPT / 'ReCogDrive-VLM-2B'},
    'recogdrive_il': {'planner': CKPT / 'ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt', 'vlm': CKPT / 'ReCogDrive-VLM-2B'},
    'diffusiondrive': {'planner': CKPT / 'DiffusionDrive/diffusiondrive_navsim_88p1_PDMS', 'backbone': CKPT / 'resnet34.a1_in1k/pytorch_model.bin',
                       'anchors': CKPT / 'DiffusionDrive/kmeans_navsim_traj_20.npy'}}
TAUS = [1., .9, .8, .7, .6, .5, .4, .3]
HORIZON = 8
SEEDS = list(range(9201, 9207))


LEAD_KS_ADD2 = (2, 4, 5, 10, 20, 50, 100)  # Addendum 2: upstream leading grids t_i = i*(100//k)


def configs_for(model, config_set='protocol'):
    """(name, grid, tau, eta); grid names a time grid (see TRAILING / DD_GRIDS). Addendum 1 of the protocol; config_set='addendum2' (ReCogDrive only) is Addendum 2."""
    if config_set == 'addendum2':
        assert model.startswith('recogdrive'), 'addendum2 config set is defined for ReCogDrive only'
        c = [(f'ddim{k}_lead_eta0_t1.0', f'lead{k}', 1., 0) for k in (2, 4, 5, 10, 20, 50)]
        c += [(f'ddim100_trail_eta0_t{t}', 'trail100', t, 0) for t in TAUS]  # 100 steps = every timestep: leading == trailing (checked)
        c += [('release5_lead_eta1_t1.0', 'lead5', 1., 1)]
        return c
    if model.startswith('recogdrive'):
        c = [(f'ddim50_trail_eta0_t{t}', 'trail50', t, 0) for t in TAUS] + [('ddim100_trail_eta0_t1.0', 'trail100', 1., 0)]
        c += [(f'ddim{k}_trail_eta0_t1.0', f'trail{k}', 1., 0) for k in [1, 2, 5, 10]]
        c += [('ddim5_lead_eta0_t1.0', 'lead5', 1., 0), ('release5_lead_eta1_t1.0', 'lead5', 1., 1)]
        return c
    return [(f'ddim9_int_eta0_t{t}', 'int9', t, 0) for t in TAUS] + [('ddim5_int_eta0_t1.0', 'int5', 1., 0), ('release2_lead_eta0_t1.0', 'rel2', 1., 0)]


def trailing_grid(k, T=100):
    """Addendum 1: t_i = round(T - i*T/k) - 1, descending; every k starts at T-1."""
    return [int(round(T - i * T / k)) - 1 for i in range(k)]


# DiffusionDrive grids: (timesteps, prev timesteps); prev=-1 means the final alpha (=1, x0), as in diffusers set_alpha_to_one.
DD_GRIDS = {'int9': (list(range(8, -1, -1)), list(range(7, -2, -1))),
            'int5': ([8, 6, 4, 3, 1], [6, 4, 3, 1, -1]),   # t_i = round(8 - i*9/5) on the same [8,0] interval
            'rel2': ([10, 0], [9, -1])}                      # release: scheduler.step from k to k-1 (upstream, set_timesteps(1000))


def grids_manifest(runner):
    if hasattr(runner, 'schedules'):return {'timesteps_descending': runner.schedules, 'spacing': 'trailN: t_i=round(100-i*100/N)-1; leadK: upstream arange(0,K)*(100//K) flipped (_init_ddim_sampler, run per K)'}
    return {k: {'timesteps': v[0], 'prev_timesteps': v[1]} for k, v in DD_GRIDS.items()}


def leading_buffers(acp, k, T=100):
    """Independent replication of upstream _init_ddim_sampler (L223-258) with ddim_eta=0, for the Addendum 2 equivalence check."""
    import torch
    t = (torch.arange(0, k) * (T // k)).long()
    a = acp[t].clone().to(torch.float32)
    ap = torch.cat([torch.tensor([1.0], dtype=torch.float32), acp[t[:-1]]])
    sig = 0.0 * ((1 - ap) / (1 - a) * (1 - a / ap)) ** 0.5
    g = {'ddim_t_schedule': t, 'ddim_t': t, 'ddim_alphas': a, 'ddim_alphas_sqrt': torch.sqrt(a), 'ddim_alphas_prev': ap,
         'ddim_sqrt_one_minus_alphas': (1.0 - a) ** 0.5, 'ddim_sigmas': sig}
    return {n: (v if n == 'ddim_t_schedule' else torch.flip(v, [0])) for n, v in g.items()}  # upstream leaves ddim_t_schedule unflipped


def bexp(x, B):
    return x.expand(B, *([-1] * (x.ndim - 1)))


def seed_noise(seed, shape):
    """Initial noise z for a seed (float32 on CPU); the generator is returned positioned just after z (per-step noise)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape, generator=g), g


# ------------------------------------------------------------------ ReCogDrive
class Recog:
    def __init__(self, model, device, cpu, config_set='protocol'):
        import torch
        repo = REPOS[model]
        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
        sys.path.insert(0, str(repo))
        self.torch, self.device, self.cpu = torch, torch.device(device), cpu
        if cpu:cpu_compat()
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
        ck = CHECKPOINTS[model]
        self.agent = ReCogDriveAgent(trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5), vlm_path=str(ck['vlm']),
            checkpoint_path=str(ck['planner']), cam_type='single', vlm_type='internvl', dit_type='small', sampling_method='ddim',
            cache_mode=False, cache_hidden_state=False, grpo=False, vlm_size='small')
        self.agent.initialize();self.agent.eval()
        self.head = self.agent.action_head
        if cpu:self.head.float()
        for p in self.agent.parameters():p.requires_grad_(False)
        self.dtype = next(self.head.parameters()).dtype
        self.config_set = config_set
        self.grids = self.snapshot_grids(lead_ks=LEAD_KS_ADD2 if config_set == 'addendum2' else (5,))
        self.check_grids()
        self.store = {};self.cfgs = [];self.seeds = [];self.seed_batch = 1;self.verify = False;self.verify_result = None
        self.head.get_action = self.patched_get_action

    def snapshot_grids(self, ks=(1, 2, 5, 10, 50, 100), lead_ks=(5,)):  # lead_ks: leading grids from the upstream init (default: only the release K=5)
        """H5: ddim_*/ddpm_* buffers for each K, release (K=5) buffers restored afterwards."""
        head, torch = self.head, self.torch
        names = [n for n in head._buffers if n.startswith(('ddpm_', 'ddim_'))]
        orig = {n: head._buffers[n] for n in names};orig_eta = head.eta;orig_steps = head.config.num_inference_steps
        grids = {};self.schedules = {};self.lead_cpu = {}
        acp = head.ddpm_alphas_cumprod.float().cpu()
        for k in sorted(set(ks) | set(lead_ks)):
            # leading grid: upstream _init_ddim_sampler (L223-258) run for this k (default only the release K=5 / ddim5_lead).
            if k in lead_ks:
                head.config.num_inference_steps = k;head._init_ddim_sampler(head.config.ddim_cfg)
                self.lead_cpu[k] = {n: head._buffers[n].clone().cpu() for n in names if n.startswith('ddim_')}
                grids[f'lead{k}'] = {n: v.to(self.device) for n, v in self.lead_cpu[k].items()}
                self.schedules[f'lead{k}'] = head._buffers['ddim_t'].tolist()
            # trailing grid: same formulas as L233-258 (alphas at t, alphas_prev = alphas at the next lower t, 1 at the lowest), flipped.
            if k not in ks:continue
            sched = torch.tensor(sorted(trailing_grid(k)), dtype=torch.long)
            a = acp[sched].clone();ap = torch.cat([torch.tensor([1.0]), acp[sched[:-1]]])
            g = {'ddim_t_schedule': sched, 'ddim_t': sched, 'ddim_alphas': a, 'ddim_alphas_sqrt': a.sqrt(), 'ddim_alphas_prev': ap,
                 'ddim_sqrt_one_minus_alphas': (1.0 - a) ** 0.5, 'ddim_sigmas': torch.zeros_like(a)}
            grids[f'trail{k}'] = {n: torch.flip(v, [0]).to(self.device) for n, v in g.items()}
            self.schedules[f'trail{k}'] = grids[f'trail{k}']['ddim_t'].tolist()
        for n in names:head._buffers[n] = orig[n]
        head.eta = orig_eta;head.config.num_inference_steps = orig_steps;head.ddim_steps = orig_steps;head.ft_denoising_steps = orig_steps
        self.release_buffers = {n: orig[n] for n in names if n.startswith('ddim_')}
        return grids

    def check_grids(self):
        """Addendum 2 equivalence checks (CPU buffers, exact): (1) the generalised upstream-init lead5 equals an independent replication
        of the upstream formulas (leading_buffers) for every k built, and the lead5 used by the protocol/release is that same grid;
        (2) leading k=100 == trailing k=100 (when both exist)."""
        torch = self.torch
        acp = self.head.ddpm_alphas_cumprod.float().cpu()
        res = {}
        for k, buf in self.lead_cpu.items():
            ref = leading_buffers(acp, k)
            assert set(ref) == set(buf), (k, set(ref) ^ set(buf))
            for n in ref:assert torch.equal(ref[n].float(), buf[n].float()), f'lead{k} buffer {n} differs from upstream formulas'
            res[f'lead{k}_equals_upstream_formulas'] = True
        if 5 in self.lead_cpu:
            assert self.schedules['lead5'] == [80, 60, 40, 20, 0]
            res['lead5_schedule_80_60_40_20_0'] = True
        if 100 in self.lead_cpu and 'trail100' in self.grids:
            for n, v in self.grids['trail100'].items():
                if n != 'ddim_t_schedule':assert torch.equal(v.cpu().float(), self.lead_cpu[100][n].float()), f'lead100 != trail100 in {n}'
            res['lead100_equals_trail100'] = True
        self.grid_checks = res
        print(json.dumps({'grid_checks': res}), flush=True)

    def embed(self, vl_features, action_input):
        """H2: verbatim prologue of get_action (L539-549)."""
        head = self.head
        vl = head.feature_encoder(vl_features)
        his = head.his_traj_encoder(action_input.his_traj.unsqueeze(1)).repeat(1, head.config.action_horizon, 1)
        ego = head.ego_status_encoder(action_input.status_feature)
        return vl, his, ego

    def ddim_loop(self, vl, his, ego, z, g_list, grid, tau, eta):
        """H3-H6. z: (B,8,3) float32 CPU; g_list: per-sample generators positioned after z (step noise)."""
        torch, head = self.torch, self.head
        B = z.shape[0];dev = self.device;dtype = self.dtype
        for n, b in self.grids[grid].items():head._buffers[n] = b
        steps = len(self.schedules[grid]);head.ddim_steps = steps
        try:
            cur = (tau * z).to(dev, dtype)
            clip = getattr(head, 'eval_randn_clip_value', 1.0);floor = getattr(head, 'eval_min_sampling_denoising_std', 0.0001)
            for i in range(steps):
                t_batch = head.make_timesteps(B, head.ddim_t[i], dev);index_batch = head.make_timesteps(B, i, dev)
                mean, logvar, _ = head.p_mean_variance(cur, t_batch, index_batch, vl, his, ego, eta == 0)
                std = torch.exp(0.5 * logvar).to(dtype)
                noise = torch.cat([torch.randn((1,) + tuple(z.shape[1:]), generator=g) for g in g_list]).to(dev, dtype)
                if eta == 0:std.zero_()
                else:std = std.clamp(min=floor)
                noise.clamp_(-clip, clip)
                cur = mean + std * noise
            fin = getattr(head, 'final_action_clip_value', 1.0)
            if fin is not None:cur.clamp_(-fin, fin)
            return head.denorm_odo(cur).float().cpu().numpy()
        finally:
            for n, b in self.release_buffers.items():head._buffers[n] = b
            head.ddim_steps = head.config.num_inference_steps

    def patched_get_action(self, vl_features, action_input, init_actions=None, deterministic=False):
        torch = self.torch
        from transformers.feature_extraction_utils import BatchFeature
        vl, his, ego = self.embed(vl_features, action_input)
        shape = (self.head.config.action_horizon, self.head.config.action_dim)
        for s0 in range(0, len(self.seeds), self.seed_batch):
            chunk = self.seeds[s0:s0 + self.seed_batch];zs = []
            for s in chunk:zs.append(seed_noise(s, (1,) + shape)[0])
            z = torch.cat(zs);B = len(chunk)
            for name, steps, tau, eta in self.cfgs:
                g_list = [seed_noise(s, (1,) + shape)[1] for s in chunk]
                out = self.ddim_loop(bexp(vl, B), bexp(his, B), bexp(ego, B), z, g_list, steps, tau, eta)
                for j, s in enumerate(chunk):self.store[(s, name)] = out[j]
        if self.verify and (self.seeds[0], 'release5_lead_eta1_t1.0') in self.store:
            torch.manual_seed(self.seeds[0])
            ref = self.head.__class__.get_action(self.head, vl_features, action_input)['pred_traj'].float().cpu().numpy()[0]
            self.verify_result = float(np.abs(ref - self.store[(self.seeds[0], 'release5_lead_eta1_t1.0')]).max())
        release = self.store.get((self.seeds[0], 'release5_lead_eta1_t1.0'))
        if release is None:release = next(iter(self.store.values()))
        return BatchFeature(data={'pred_traj': torch.from_numpy(release)[None].to(self.device)})

    def run_scene(self, agent_input):
        self.store = {}
        with self.torch.no_grad():self.agent.compute_trajectory(agent_input)
        return self.store


def cpu_compat():
    """Neutralize upstream hard-coded CUDA use for the CPU smoke test (runtime only, nothing edited on disk)."""
    import torch, types
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    sys.path.insert(0, str(REPOS['recogdrive_rl']))
    import navsim.agents.recogdrive.recogdrive_backbone as bb
    import navsim.agents.recogdrive.recogdrive_agent as ag
    class TorchProxy(types.ModuleType):
        def __getattr__(self, n):return getattr(torch, n)
        def device(self, *a, **k):return torch.device('cpu')
    bb.torch = TorchProxy('torch')
    orig = ag.RecogDriveBackbone
    ag.RecogDriveBackbone = lambda **kw: orig(**{**kw, 'device': 'cpu'})


# ------------------------------------------------------------------ DiffusionDrive v1
class DD:
    def __init__(self, device, cpu):
        import torch
        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
        sys.path.insert(0, str(REPOS['diffusiondrive']))
        from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        ck = CHECKPOINTS['diffusiondrive']
        cfg = TransfuserConfig(bkb_path=str(ck['backbone']), plan_anchor_path=str(ck['anchors']))
        self.torch, self.device = torch, torch.device(device)
        self.agent = TransfuserAgent(cfg, lr=6e-4, checkpoint_path=str(ck['planner']))
        self.agent.initialize();self.agent.eval().to(self.device)
        for p in self.agent.parameters():p.requires_grad_(False)
        self.head = self.agent._transfuser_model._trajectory_head
        self.original_forward_test = self.head.forward_test
        self.head.forward_test = self.patched_forward_test
        self.store = {};self.cfgs = [];self.seeds = [];self.seed_batch = 1;self.verify = False;self.verify_result = None

    def dd_loop(self, rep, z, grid, tau, use_scheduler_step=False):
        """H8-H10: reproduction of forward_test L505-558 with step_num=steps and the noise tau*z. rep: forward_test args, batch B."""
        torch, head = self.torch, self.head
        ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img = rep
        bs = ego_query.shape[0];device = ego_query.device
        head.diffusion_scheduler.set_timesteps(1000, device)
        ts, prevs = DD_GRIDS[grid]
        roll = torch.tensor(ts, dtype=torch.long, device=device)
        acp = head.diffusion_scheduler.alphas_cumprod.to(device).float();final = head.diffusion_scheduler.final_alpha_cumprod.to(device).float()
        img = head.norm_odo(head.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1))
        noise = (tau * z).to(device, img.dtype)
        trunc = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = head.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc)
        mode = img.shape[1]
        for k, prev in zip(roll, prevs):
            noisy = head.denorm_odo(torch.clamp(img, min=-1, max=1))
            from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position
            emb = gen_sineembed_for_position(noisy, hidden_dim=64).flatten(-2)
            traj_feature = head.plan_anchor_encoder(emb).view(bs, mode, -1)
            timesteps = k[None].to(device).expand(bs)
            time_embed = head.time_mlp(timesteps).view(bs, 1, -1)
            reg_list, cls_list = head.diff_decoder(traj_feature, noisy, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding, global_img)
            reg, cls = reg_list[-1], cls_list[-1]
            x_start = head.norm_odo(reg[..., :2])
            if use_scheduler_step:
                img = head.diffusion_scheduler.step(model_output=x_start, timestep=k, sample=img).prev_sample
            else:  # DDIM eta=0, prediction_type=sample, same update as diffusers DDIMScheduler.step but with an explicit previous timestep
                a_t = acp[k];a_p = acp[prev] if prev >= 0 else final
                x0 = x_start.clamp(-1, 1) if head.diffusion_scheduler.config.clip_sample else x_start
                eps = (img - a_t.sqrt() * x0) / (1 - a_t).sqrt()
                img = a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps
        idx = cls.argmax(dim=-1)[..., None, None, None].repeat(1, 1, head._num_poses, 3)
        return torch.gather(reg, 1, idx).squeeze(1)

    def patched_forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img):
        torch = self.torch;assert ego_query.shape[0] == 1
        shape = (1,) + tuple(self.head.plan_anchor.shape)
        for s0 in range(0, len(self.seeds), self.seed_batch):
            chunk = self.seeds[s0:s0 + self.seed_batch];B = len(chunk)
            z = torch.cat([seed_noise(s, shape)[0] for s in chunk])
            rep = (bexp(ego_query, B), bexp(agents_query, B), bexp(bev_feature, B), bev_spatial_shape, bexp(status_encoding, B), global_img)
            for name, grid, tau, eta in self.cfgs:
                out = self.dd_loop(rep, z, grid, tau, use_scheduler_step=(grid == 'rel2')).float().cpu().numpy()
                for j, s in enumerate(chunk):self.store[(s, name)] = out[j]
        if self.verify and (self.seeds[0], 'release2_lead_eta0_t1.0') in self.store:
            torch.manual_seed(self.seeds[0])
            ref = self.original_forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img)['trajectory'].float().cpu().numpy()[0]
            self.verify_result = float(np.abs(ref - self.store[(self.seeds[0], 'release2_lead_eta0_t1.0')]).max())
            # the explicit-prev DDIM update must equal DDIMScheduler.step on the all-integer 9-step grid
            zz = seed_noise(self.seeds[0], shape)[0];rr = (ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img)
            self.verify_update_maxabs = float((self.dd_loop(rr, zz, 'int9', 1., True) - self.dd_loop(rr, zz, 'int9', 1., False)).abs().max())
        out = next(iter(self.store.values()))
        return {'trajectory': torch.from_numpy(out)[None].to(self.device)}

    def run_scene(self, agent_input):
        self.store = {};torch = self.torch
        feats = {}
        for b in self.agent.get_feature_builders():feats.update(b.compute_features(agent_input))
        feats = {k: v.unsqueeze(0).to(self.device) for k, v in feats.items()}
        with torch.no_grad():self.agent.forward(feats)
        return self.store


# ------------------------------------------------------------------ scenes / main
def load_scenes(args, sensor_config, agent_cls_has_paths):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import inspect
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    data = Path(os.environ['OPENSCENE_DATA_ROOT'])
    spec = json.loads(args.scenes.read_text())['scenes'][args.range[0]:args.range[1]]
    cfg = OmegaConf.load(REPOS[args.model] / 'navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml')
    def loader(sensors, **kw):
        flt = instantiate(cfg);flt.log_names = sorted({s['log'] for s in spec});flt.tokens = [s['token'] for s in spec]
        return SceneLoader(sensor_blobs_path=data / 'sensor_blobs/test', data_path=data / 'navsim_logs/test', scene_filter=flt, sensor_config=sensors, **kw)
    kw = {'load_image_path': True} if 'load_image_path' in inspect.signature(SceneLoader.__init__).parameters else {}
    return spec, loader(sensor_config, **kw), loader(SensorConfig.build_no_sensors())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['recogdrive_rl', 'recogdrive_il', 'diffusiondrive'], required=True)
    p.add_argument('--scenes', type=Path, default=ARTIFACTS / 'navsim_scenes_v1_20261028.json');p.add_argument('--out', type=Path, required=True)
    p.add_argument('--range', type=int, nargs=2, default=[0, 500]);p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--config-set', choices=['protocol', 'addendum2'], default='protocol', help='addendum2: leading-grid sweep + 100-step reference (ReCogDrive only)')
    p.add_argument('--configs', nargs='+', default=None, help='subset of config names (smoke tests only; the protocol grid is the default)')
    p.add_argument('--seed-batch', type=int, default=6, help='seeds integrated together as one batch (1 = one at a time)')
    p.add_argument('--verify-release', action='store_true', help='compare our release integrator with upstream under the same CPU RNG (CPU only)')
    args = p.parse_args()
    import torch
    cpu = not torch.cuda.is_available();device = 'cpu' if cpu else 'cuda:0'
    torch.set_num_threads(int(os.environ.get('FROST_THREADS', '8' if cpu else '2')))
    allc = configs_for(args.model, args.config_set);byname = {c[0]: c for c in allc}
    cfgs = allc if not args.configs else [byname[n] for n in args.configs]
    args.out.mkdir(parents=True, exist_ok=args.resume)
    source = Path(__file__).read_bytes();(args.out / 'source_snapshot.py').write_bytes(source)
    runner = Recog(args.model, device, cpu, args.config_set) if args.model.startswith('recogdrive') else DD(device, cpu)
    runner.cfgs, runner.seeds, runner.seed_batch, runner.verify = cfgs, args.seeds, args.seed_batch, args.verify_release
    spec, loader, human_loader = load_scenes(args, runner.agent.get_sensor_config(), True)
    plans = {};timing = []
    if args.resume and (args.out / 'plans.npz').exists():
        with np.load(args.out / 'plans.npz') as z:plans = {k: z[k] for k in z.files}
        if (args.out / 'timing.json').exists():timing = json.loads((args.out / 'timing.json').read_text())
    done = {k[:-len('_human')] for k in plans if k.endswith('_human')}
    for s in spec:
        tok = s['token']
        if tok in done:continue
        t0 = time.time()
        store = runner.run_scene(loader.get_agent_input_from_token(tok));t1 = time.time()
        for (seed, name), traj in store.items():plans[f'{tok}_s{seed}_{name}'] = traj.astype(np.float32).reshape(HORIZON, 3)
        plans[f'{tok}_human'] = np.asarray(human_loader.get_scene_from_token(tok).get_future_trajectory(HORIZON).poses, np.float32)
        timing.append({'token': tok, 'seconds': t1 - t0, 'verify_release_maxabs': runner.verify_result})
        np.savez_compressed(args.out / 'plans.npz', **plans);(args.out / 'timing.json').write_text(json.dumps(timing))
        print(json.dumps({'scene': tok, 'plans': len(plans), 'seconds': round(t1 - t0, 1), 'verify_release_maxabs': runner.verify_result, 'verify_dd_update_maxabs': getattr(runner, 'verify_update_maxabs', None)}), flush=True)
    (args.out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'model': args.model, 'device': device,
        'cpu_smoke_float32_head': cpu, 'scenes_file': str(args.scenes), 'scenes_sha256': hashlib.sha256(args.scenes.read_bytes()).hexdigest(),
        'config_set': args.config_set, 'grid_checks': getattr(runner, 'grid_checks', None), 'range': args.range, 'n_scenes': len(spec), 'seeds': args.seeds, 'configs': cfgs, 'config_fields': ['name', 'grid', 'tau', 'eta'], 'grids': grids_manifest(runner),
        'source_sha256': hashlib.sha256(source).hexdigest(), 'checkpoints': {k: str(v) for k, v in CHECKPOINTS[args.model].items()},
        'repo': str(REPOS[args.model]), 'torch': torch.__version__, 'seed_batch': args.seed_batch,
        'limitations': ['Noise z drawn per seed from a CPU torch.Generator (seed) and shared by all configs; per-step noise of the release comes from the same generator after z (so the release plan is not bit-identical to an upstream CUDA run).',
            'Addendum 1: ReCogDrive eta=0 sweep uses trailing spacing (K=1 -> t=99); ddim5_lead_eta0 and the release use the upstream leading grid.',
            'Addendum 1: DiffusionDrive reference = DDIM eta=0 over integer timesteps 8..0 (9 steps), convergence check 5 steps on [8,0] (t=8,6,4,3,1, last step to x0).']},
        indent=2))


if __name__ == '__main__':main()
