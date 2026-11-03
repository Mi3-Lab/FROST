"""Adaptive reference solver (scipy RK45 = Dormand-Prince 5(4)) for flow-ODE dx/dsigma = v(x, sigma) from sigma0 down to 0, with an NFE budget.
Used by the SimWAM and Cosmos 3 Nano convergence extensions (diagnostics only)."""
import numpy as np
from scipy.integrate import solve_ivp


class Budget(Exception):
    pass


def rk45(f, y0, sigma0, rtol, atol, max_nfe):
    """f(sigma, y_flat) -> dy/dsigma (flat). Returns dict(y, nfe, ok, steps, reason)."""
    n = [0]

    def g(s, y):
        n[0] += 1
        if n[0] > max_nfe: raise Budget()
        return f(s, y)
    try:
        r = solve_ivp(g, (sigma0, 0.0), np.asarray(y0, np.float64).ravel(), method='RK45', rtol=rtol, atol=atol)
        return dict(y=r.y[:, -1], nfe=int(n[0]), ok=bool(r.success), steps=int(len(r.t) - 1), reason=r.message)
    except Budget:
        return dict(y=None, nfe=int(n[0]), ok=False, steps=None, reason=f'budget {max_nfe} NFE exceeded')


# ---- Dormand-Prince 5(4) with joint (vision, action) state for the Cosmos pipeline; step control on the action state only.
_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]


def dp_generator(xv, xa, sigma0, rtol, atol, max_nfe, valid, info):
    """Generator driving an adaptive DP5(4) solve of dx/dsigma = v from sigma0 to 0. `yield (sigma, xv, xa)` asks for the velocity there and
    receives (vv, va). The first request (the initial state) is assumed already evaluated and is answered by the caller. Error control uses
    action channels [:valid] only. Fills info with nfe/steps/rejected/ok."""
    import torch
    sigma = float(sigma0); h = 0.02; nfe = 0; steps = 0; rej = 0
    k1 = yield (sigma, xv, xa); nfe += 1
    ok = True
    while sigma > 1e-9:
        h = min(h, sigma); ks = [k1]
        for s in range(1, 7):
            xv_s = xv - h * sum(a * k[0] for a, k in zip(_A[s], ks) if a); xa_s = xa - h * sum(a * k[1] for a, k in zip(_A[s], ks) if a)
            if nfe >= max_nfe: ok = False; break
            ks.append((yield (sigma - _C[s] * h, xv_s, xa_s))); nfe += 1
        if not ok: break
        xv5 = xv - h * sum(b * k[0] for b, k in zip(_B5, ks) if b); xa5 = xa - h * sum(b * k[1] for b, k in zip(_B5, ks) if b)
        err = (h * sum((b5 - b4) * k[1][..., :valid] for b5, b4, k in zip(_B5, _B4, ks) if b5 - b4)).abs()
        scale = atol + rtol * torch.maximum(xa[..., :valid].abs(), xa5[..., :valid].abs())
        e = float(((err / scale) ** 2).mean().sqrt())
        fac = min(5.0, max(0.2, 0.9 * (e + 1e-16) ** -0.2))
        if e <= 1.0:
            sigma -= h; xv, xa = xv5, xa5; steps += 1; k1 = ks[6]
            if sigma <= 1e-9: break
        else:
            rej += 1
        h *= fac
    info.update(nfe=nfe, steps=steps, rejected=rej, ok=ok)
    return xv, xa


class _Handle:
    def __init__(self, ctl, which): self.ctl, self.which = ctl, which
    order = 1
    class config: num_train_timesteps = 1000
    def step(self, v, t, x, return_dict=False): return (self.ctl._step(self.which, v, x),)


class AdaptiveController:
    """Duck-typed pipeline scheduler running an adaptive solve: one pipeline iteration = one model evaluation. Vision and action share the step
    sizes; the pipeline deep-copies its scheduler for the action stream, so __deepcopy__ returns the action handle."""
    order = 1
    class config: num_train_timesteps = 1000

    def __init__(self, rtol, atol, max_nfe, valid=9, sigma0=1 - 1e-3):
        self.rtol, self.atol, self.max_nfe, self.valid, self.sigma0 = rtol, atol, max_nfe, valid, sigma0
        self.info = {}; self.gen = None; self.cur = sigma0; self.done = False; self.vis = None; self.xv0 = None; self.bv = self.ba = None
        self.timesteps = self; self.sigmas = self
        self.action_handle = _Handle(self, 'a')

    # scheduler interface used by the pipeline
    def set_timesteps(self, num_inference_steps, device=None, sigmas=None): self.cur = self.sigma0
    def __deepcopy__(self, memo): return self.action_handle
    def __len__(self): return self.max_nfe + 1
    def __getitem__(self, i): return self.cur
    def __iter__(self):
        import torch
        yield torch.tensor(self.sigma0 * 1000.0, dtype=torch.float32)
        while not self.done:
            yield torch.tensor(self.cur * 1000.0, dtype=torch.float32)
    def step(self, v, t, x, return_dict=False): return (self._step('v', v, x),)

    def _step(self, which, v, x):
        import torch
        v = v[0].float()
        if which == 'v':
            if self.vis is None:
                self.xv0 = x[0].float().clone(); self.bv = x.clone().float()
            self.vis = v; return self.bv
        # action stream: both velocities for the current request are now available
        if self.gen is None:
            self.ba = x.clone().float(); xa0 = x[0].float().clone()[None]
            self.gen = dp_generator(self.xv0[None], xa0, self.sigma0, self.rtol, self.atol, self.max_nfe, self.valid, self.info)
            next(self.gen)
        try:
            sig, xv, xa = self.gen.send((self.vis[None], v[None]))
            self.cur = sig
        except StopIteration as e:
            xv, xa = e.value; self.done = True
        self.bv[0].copy_(xv[0]); self.ba[0].copy_(xa[0]); self.vis = None
        return self.ba
