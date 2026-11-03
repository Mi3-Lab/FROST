"""Compiled Alpamayo-1.5 pipeline on the Thor: images + prompt -> reasoning text -> prefix KV -> trajectory.

  inputs  (host, PyTorch/HF code, no weights): chat template, image preprocessing, history-trajectory tokens,
          3-D mRoPE position ids (HF get_rope_index), vision grid inputs (HF pos-embed interpolation and 2-D RoPE).
  vision  : VisionGraph engine (or the torch graph)            -> image embeds + 3 deepstack features
  prefill : TextChunkGraph engines, profile 0, on the prompt   -> KV cache rows [0, n_prompt), last hidden
  decode  : TextChunkGraph engines, profile 1, one token/step  -> KV row j; HeadGraph -> logits; HF sampling
            (same LogitsProcessorList as Alpamayo's vlm.generate: trajectory tokens masked, temperature, top-p;
            torch.multinomial on the GPU after torch.manual_seed(123), stop one token after <|traj_future_start|>)
  handoff : the cache tensor [36,2,8,C,128] bf16 IS the past_kv input of the action-expert engines of
            deploy/thor/trt (zero copy); key_bias masks positions >= offset (after <|traj_future_start|>);
            RoPE of the 64 action tokens from rope_deltas + offset, exactly as Alpamayo1_5._build_expert_pos_ids_and_attn_mask.

Backends: TorchBackend (the graphs of vlm_graph.py in PyTorch, for checking the graph code) and TRTBackend (engines).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from vlm_graph import (BF16, NEG, HeadGraph, TextChunkGraph, VisionGraph, _alpamayo_imports, checkpoint_path,
                       load_embedding, load_head, load_text_layers, load_vision, text_cos_sin, vision_aux, vlm_config)

CAMERAS = [0, 1, 2, 6]
N_LAYERS = 36


# ------------------------------------------------------------------------------------------------ host-side inputs
class Inputs:
    """Everything on the host side that the PyTorch pipeline computes before the VLM forward. No weights loaded
    except the (small) vision position-embedding table for the grid inputs."""

    def __init__(self, checkpoint=None, device='cuda'):
        from accelerate import init_empty_weights
        Alpamayo1_5, _ = _alpamayo_imports()
        from alpamayo1_5 import helper
        from alpamayo1_5.models.token_utils import to_special_token
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
        self.cfg, self.vcfg = vlm_config(checkpoint)
        old = torch.get_default_dtype(); torch.set_default_dtype(BF16)
        try:
            with init_empty_weights(): self.skel = Alpamayo1_5(self.cfg)
        finally: torch.set_default_dtype(old)
        self.helper, self.tok = helper, self.skel.tokenizer
        self.processor = helper.get_processor(self.tok)
        self.device = device
        self.eos_traj = self.tok.convert_tokens_to_ids(to_special_token('traj_future_start'))
        self.rotary = Qwen3VLTextRotaryEmbedding(self.vcfg.text_config).to(device)
        self._grid_cache = {}
        self._visual_for_aux = None
        self.checkpoint = checkpoint

    @staticmethod
    def load_clip(path):
        """Decode the stored clip (np.load of a compressed npz: ~190 ms of zlib, a property of the test files, not of a car,
        where camera frames arrive as raw buffers) into uint8 frames [4,4,3,H,W] and the ego history."""
        with np.load(path) as z:
            return {'images': torch.from_numpy(z['image_frames']).permute(0, 1, 4, 2, 3).contiguous(),
                    'hx': torch.from_numpy(z['ego_history_xyz']), 'hr': torch.from_numpy(z['ego_history_rot'])}

    def tokenize(self, src, gpu=False):
        """src: a clip path or the dict of load_clip. gpu=True runs the image resize / normalise / patchify of the HF fast
        processor on the GPU (same code, tensors on cuda; pixel values differ from the CPU path by at most 1/64 of an
        8-bit step after normalisation, rel. L2 0.46%): ~123 -> ~26 ms for 16 frames of 1080x1920."""
        c = src if isinstance(src, dict) else self.load_clip(src)
        images = c['images']; hx = c['hx'][None, None].to(self.device); hr = c['hr'][None, None].to(self.device)
        assert images.shape[:2] == (4, 4) and hx.shape[-2] == 16
        im = images.flatten(0, 1)
        if gpu: im = im.to(self.device, non_blocking=True)
        messages = self.helper.create_message(im, camera_indices=torch.tensor(CAMERAS))
        tk = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                                return_dict=True, return_tensors='pt', min_pixels=196608, max_pixels=196608,
                                                **({'device': self.device} if gpu else {}))
        ids = self.skel.fuse_traj_tokens(tk['input_ids'].to(self.device), {'ego_history_xyz': hx, 'ego_history_rot': hr})
        grid = tk['image_grid_thw']
        pos_ids, deltas = self.skel.vlm.model.get_rope_index(ids.cpu(), grid, None, attention_mask=tk['attention_mask'])
        return {'input_ids': ids, 'pixel_values': tk['pixel_values'].to(self.device), 'grid': grid,
                'position_ids': pos_ids.to(self.device), 'rope_deltas': int(deltas.view(-1)[0]), 'hx': hx, 'hr': hr}

    def grid_inputs(self, grid):
        key = tuple(map(tuple, grid.tolist()))
        if key not in self._grid_cache:
            if self._visual_for_aux is None:   # HF vision module on CPU for its pos-embed table (other weights unused)
                self._visual_for_aux = load_vision(self.checkpoint, device='cpu')
            pos, cos, sin = vision_aux(self._visual_for_aux, grid)
            self._grid_cache[key] = (pos.to(self.device), cos.to(self.device), sin.to(self.device))
        return self._grid_cache[key]

    def text_rope(self, position_ids):
        return text_cos_sin(self.rotary, position_ids)

    def expert_rope(self, offset, rope_deltas, n=64):
        """cos/sin [n,128] fp32 of the action tokens (Alpamayo1_5._build_expert_pos_ids_and_attn_mask)."""
        pos = (torch.arange(n, device=self.device) + rope_deltas + offset).view(1, 1, n).expand(3, 1, n)
        dummy = torch.zeros(1, dtype=torch.float32, device=self.device)
        cos, sin = self.rotary(dummy, pos)
        return cos[0].float(), sin[0].float()

    def logits_processor(self, input_ids, top_p=.98, temperature=.6):
        """The LogitsProcessorList that vlm.generate builds in Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout."""
        import copy
        from transformers import LogitsProcessorList
        from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
        vlm = self.skel.vlm
        gc = copy.deepcopy(vlm.generation_config)
        gc.top_p, gc.temperature, gc.do_sample, gc.num_return_sequences = top_p, temperature, True, 1
        gc.max_new_tokens, gc.top_k, gc.pad_token_id = 256, None, self.tok.pad_token_id
        lp = LogitsProcessorList([ExpertLogitsProcessor(self.cfg.traj_token_start_idx, self.cfg.traj_vocab_size)])
        procs = vlm._get_logits_processor(generation_config=gc, input_ids_seq_length=input_ids.shape[1], encoder_input_ids=input_ids,
                                          prefix_allowed_tokens_fn=None, logits_processor=lp, device=str(input_ids.device), model_kwargs={})
        eos = gc.eos_token_id; eos = [eos] if isinstance(eos, int) else list(eos or [])
        return procs, eos


# ------------------------------------------------------------------------------------------------ backends
def chunk_bounds(n_chunks, n_layers=N_LAYERS):
    step = n_layers // n_chunks; assert step * n_chunks == n_layers
    return [(i * step, (i + 1) * step) for i in range(n_chunks)]


class TorchBackend:
    """The vlm_graph modules in PyTorch on the GPU (checks the graph code against the HF reference)."""

    def __init__(self, n_chunks=4, checkpoint=None, device='cuda', ppi=720, sdpa=True):
        self.ppi = ppi                                          # patches per image: 20 x 36 at 196608 px (16 px patches)
        import vlm_graph; vlm_graph.USE_SDPA = sdpa
        self.vis = VisionGraph(load_vision(checkpoint, device), ppi).eval()
        self.chunks = []
        for a, b in chunk_bounds(n_chunks):
            layers, tc = load_text_layers(a, b, checkpoint, device)
            self.chunks.append(TextChunkGraph(layers, tc, a).eval())
        norm, head = load_head(checkpoint, device)
        self.head_g = HeadGraph(norm, head).eval()
        self.bounds = chunk_bounds(n_chunks)

    @torch.no_grad()
    def vision(self, pix, pos, cos, sin):
        return self.vis(pix, pos, cos, sin)

    @torch.no_grad()
    def chunk(self, i, h, past, bias, cos, sin, deep, mode):
        g = self.chunks[i]
        return g(h, past, bias, cos, sin, deep if g.has_deepstack else None)

    @torch.no_grad()
    def head(self, h):
        return self.head_g(h)


# ------------------------------------------------------------------------------------------------ generation
class Pipeline:
    def __init__(self, backend, inputs, capacity=3456, embed_device='cuda', graph=False, share=None):
        self.be, self.inp, self.C, self.use_graph = backend, inputs, capacity, graph
        self.graph = None
        if share is not None:   # reuse another pipeline's embedding table and KV cache (same memory)
            self.embed, self.cache = share.embed, share.cache
        else:
            self.embed = load_embedding(inputs.checkpoint).to(embed_device)
            self.cache = torch.zeros(N_LAYERS, 2, 8, capacity, 128, dtype=BF16, device='cuda')
        self.dummy = {}
        self.zdeep = torch.zeros(3, 1, inputs.vcfg.text_config.hidden_size, dtype=BF16, device='cuda')   # no image token in decode
        self.hidden = inputs.vcfg.text_config.hidden_size

    def _embeds(self, ids):
        return self.embed[ids.to(self.embed.device)].to('cuda')

    def prefill(self, x, bucket=None):
        """x = Inputs.tokenize(...). Writes cache rows [0, n) and returns the last hidden state [1,1,4096] and timings."""
        t = {}
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ids = x['input_ids']; n = ids.shape[1]; S = bucket or n; assert S >= n
        pos, vcos, vsin = self.inp.grid_inputs(x['grid'])
        assert all(int(t * hh * ww) == self.be.ppi for t, hh, ww in x['grid'].tolist()), (x['grid'], self.be.ppi)
        ev[0].record()
        img, deep = self.be.vision(x['pixel_values'].float(), pos, vcos, vsin)
        ev[1].record()
        h = torch.zeros(1, S, self.hidden, dtype=BF16, device='cuda')
        h[0, :n] = self._embeds(ids[0])
        mask = (ids[0] == self.inp.vcfg.image_token_id)
        assert int(mask.sum()) == img.shape[0], (int(mask.sum()), img.shape)
        h[0, :n][mask] = img
        dfull = torch.zeros(3, S, self.hidden, dtype=BF16, device='cuda'); dfull[:, :n][:, mask] = deep
        pid = torch.zeros(3, 1, S, dtype=torch.long, device='cuda'); pid[:, :, :n] = x['position_ids']
        if S > n: pid[:, :, n:] = pid[:, :, n - 1:n] + torch.arange(1, S - n + 1, device='cuda')
        cos, sin = self.inp.text_rope(pid)
        Pd = getattr(self.be, 'prefill_past', 1)              # masked dummy past slots of the prefill profile
        bias = torch.full((S, Pd + S), NEG, device='cuda')
        bias[:, Pd:] = torch.triu(torch.full((S, S), NEG, device='cuda'), diagonal=1)
        for i, (a, b) in enumerate(self.be.bounds):
            past = self.dummy.setdefault((a, b), torch.zeros(b - a, 2, 8, Pd, 128, dtype=BF16, device='cuda'))
            h, new = self.be.chunk(i, h, past, bias, cos, sin, dfull, 'prefill')
            self.cache[a:b, :, :, :n] = new[:, :, :, :n]
        ev[2].record(); torch.cuda.synchronize()
        t['vision_ms'] = ev[0].elapsed_time(ev[1]); t['prefill_ms'] = ev[1].elapsed_time(ev[2])
        return h[:, n - 1:n], t

    def decode_step(self, token, j, rope_deltas):
        """Feed one token at absolute position j (cache rows [0, j) valid); writes row j; returns logits [1,V] bf16."""
        h = self._embeds(token.view(1)).view(1, 1, self.hidden)
        p = torch.full((3, 1, 1), j + rope_deltas, dtype=torch.long, device='cuda')
        cos, sin = self.inp.text_rope(p)
        bias = torch.zeros(1, self.C + 1, device='cuda'); bias[0, j:self.C] = NEG
        for i, (a, b) in enumerate(self.be.bounds):
            h, new = self.be.chunk(i, h, self.cache[a:b], bias, cos, sin, self.zdeep, 'decode')
            self.cache[a:b, :, :, j] = new[:, :, :, 0]
        return self.be.head(h)

    # ---- CUDA graph of one decode step (token and position live in device buffers) ----
    def _graph_step(self):
        h = self.embed[self.g_tok].view(1, 1, self.hidden)
        p = (self.g_j + self.g_rd).view(1, 1, 1).expand(3, 1, 1)
        cos, sin = self.inp.text_rope(p)
        bias = torch.where((self.g_ar < self.g_j) | (self.g_ar == self.C), 0., NEG).view(1, self.C + 1)
        for i, (a, b) in enumerate(self.be.bounds):
            h, new = self.be.chunk(i, h, self.cache[a:b], bias, cos, sin, self.zdeep, 'decode')
            self.cache[a:b].index_copy_(3, self.g_j, new)
        return self.be.head(h)

    def _capture(self):
        assert self.embed.is_cuda, 'graph decode needs the embedding table on the GPU'
        self.g_tok = torch.zeros(1, dtype=torch.long, device='cuda'); self.g_j = torch.full((1,), self.C - 1, dtype=torch.long, device='cuda')
        self.g_rd = torch.zeros(1, dtype=torch.long, device='cuda'); self.g_ar = torch.arange(self.C + 1, device='cuda')
        saved = self.cache[:, :, :, self.C - 1].clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2): self._graph_step()          # warm-up (shapes, TRT first-enqueue work) outside the capture
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.g_logits = self._graph_step()
        self.cache[:, :, :, self.C - 1] = saved

    def decode_step_graph(self, token, j, rope_deltas):
        if self.graph is None: self._capture()
        self.g_tok.copy_(token.view(1)); self.g_j.fill_(j); self.g_rd.fill_(rope_deltas)
        self.graph.replay()
        return self.g_logits

    @torch.no_grad()
    def run(self, x, seed=123, max_new=256, forced=None, top_p=.98, temperature=.6):
        """Reasoning rollout. forced = reference generated token ids (teacher forcing: those tokens are fed, while the
        token sampled at each step with the same RNG stream is recorded for agreement)."""
        ids = x['input_ids']; n = ids.shape[1]
        assert n + max_new <= self.C
        procs, eos = self.inp.logits_processor(ids, top_p, temperature)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        hlast, tm = self.prefill(x, getattr(self.be, 'prefill_bucket', None))
        logits = self.be.head(hlast)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        seq = ids.clone(); sampled, fed = [], []
        t1 = time.perf_counter(); stop_next = False; steps = 0
        while True:
            scores = procs(seq, logits.float())
            nxt = torch.multinomial(torch.softmax(scores, dim=-1), 1)[0]
            sampled.append(int(nxt))
            tok = nxt if forced is None else torch.tensor([forced[len(fed)]], device='cuda')
            fed.append(int(tok)); seq = torch.cat([seq, tok.view(1, 1)], 1)
            if stop_next or int(tok) in eos or len(fed) >= max_new or (forced is not None and len(fed) >= len(forced)): break
            if int(tok) == self.inp.eos_traj: stop_next = True
            step = self.decode_step_graph if self.use_graph else self.decode_step
            logits = step(tok, n + len(fed) - 1, x['rope_deltas']); steps += 1
        torch.cuda.synchronize(); t2 = time.perf_counter()
        P = n + len(fed) - 1                                  # the last token is never fed (HF generate)
        gen = torch.tensor(fed)
        hits = (gen == self.inp.eos_traj).nonzero()
        offset = n + (int(hits[0]) + 1 if len(hits) else len(fed))
        tm.update({'decode_ms': (t2 - t1) * 1e3, 'decode_steps': steps, 'total_vlm_ms': (t2 - t0) * 1e3,
                   'decode_ms_per_token': (t2 - t1) * 1e3 / max(steps, 1)})
        return {'P': P, 'n_prompt': n, 'offset': offset, 'gen': fed, 'sampled': sampled,
                'text': self.inp.tok.decode(fed, skip_special_tokens=False), 'timing': tm}

    def field_inputs(self, r, rope_deltas):
        """key_bias [C] fp32, cos/sin [64,128] fp32 for the action-expert engine (cache = past_kv, zero copy)."""
        bias = torch.full((self.C,), NEG, device='cuda'); bias[:r['offset']] = 0
        cos, sin = self.inp.expert_rope(r['offset'], rope_deltas)
        return bias, cos, sin
