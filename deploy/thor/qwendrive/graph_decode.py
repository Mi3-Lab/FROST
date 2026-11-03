"""Qwen-Drive reasoning prefill + greedy decode on a HF StaticCache (hybrid GDN/full attention) with the one-token step in a CUDA graph.
Drop-in for QwenDriveForPlanning._prefill_with_reasoning: returns (scene_cache list [B, L, kv, hd], anchor [3,1], text)."""
import time
import torch
import thor_patches  # noqa


class GraphDecoder:
    def __init__(self, model, max_len=4700):
        from transformers import StaticCache
        self.m = model; self.vlm = model.vlm; self.max_len = max_len
        self.cache = StaticCache(config=self.vlm.config.get_text_config(), max_cache_len=max_len)
        self.tok = torch.zeros(1, 1, dtype=torch.long, device='cuda'); self.cp = torch.zeros(1, dtype=torch.long, device='cuda')
        self.rd = torch.zeros((), dtype=torch.long, device='cuda'); self.pos = torch.zeros(3, 1, 1, dtype=torch.long, device='cuda')
        self.sup = torch.zeros(1, self.vlm.config.get_text_config().vocab_size, device='cuda', dtype=torch.bfloat16)
        self.graph = None; self.new_tok = None
        p = model.processor; self.terms = [p.im_end_id, model.config.vlm_config.text_config.eos_token_id]
        self.min_new = int(getattr(model.config, 'min_reasoning_tokens', 0))

    def _step(self):
        self.pos.copy_((self.cp + self.rd).view(1, 1, 1).expand(3, 1, 1))
        o = self.vlm(input_ids=self.tok, past_key_values=self.cache, use_cache=True, cache_position=self.cp, position_ids=self.pos, logits_to_keep=1)
        return (o.logits[:, -1] + self.sup).argmax(-1, keepdim=True)

    def _prefill(self, inputs):
        ids = inputs['input_ids']; P = ids.shape[1]
        self.cache.reset()
        o = self.vlm(input_ids=ids, pixel_values=inputs['pixel_values'], image_grid_thw=inputs['image_grid_thw'], mm_token_type_ids=self.m._modality_ids(ids),
                     past_key_values=self.cache, use_cache=True, cache_position=torch.arange(P, device='cuda'), logits_to_keep=1)
        self.rd.copy_(self.vlm.model.rope_deltas.reshape(-1)[0]); self.cp.fill_(P)
        return o.logits[:, -1], P

    @torch.no_grad()
    def capture(self, inputs):
        logits, P = self._prefill(inputs); self.tok.copy_(logits.argmax(-1, keepdim=True))
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2): self._step()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        logits, P = self._prefill(inputs); self.tok.copy_(logits.argmax(-1, keepdim=True))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.new_tok = self._step()

    @torch.no_grad()
    def run(self, inputs, max_new):
        m, P_ = self.m, inputs['input_ids'].shape[1]
        prompt_anchor = m._rope_positions(inputs['input_ids'], inputs['image_grid_thw'])[:, :, -1]
        t0 = time.perf_counter(); torch.cuda.synchronize()
        logits, P = self._prefill(inputs); torch.cuda.synchronize(); t1 = time.perf_counter()
        self.sup.zero_()
        if self.min_new > 0: self.sup[0, self.terms] = float('-inf')
        first = (logits + self.sup).argmax(-1, keepdim=True)
        gen = [int(first)]; self.tok.copy_(first)
        while True:
            if gen[-1] in self.terms or len(gen) >= max_new: break
            if len(gen) == self.min_new: self.sup.zero_()
            self.graph.replay(); gen.append(int(self.new_tok)); self.tok.copy_(self.new_tok); self.cp += 1
        t2 = time.perf_counter()
        content = gen[:gen.index(next(t for t in gen if t in self.terms))] if any(t in self.terms for t in gen) else gen
        proc = m.processor
        from qwen_drive.modeling_qwen_drive import _strip_thinking
        text = _strip_thinking(proc.tokenizer.decode(content, skip_special_tokens=True))
        closed = content + [proc.im_end_id] + proc.newline_ids
        fed = len(gen) - 1                       # tokens already in the cache after the prompt (the last generated one is not fed)
        pending = closed[fed:]
        if pending:
            S = len(pending); cpos = torch.arange(P + fed, P + fed + S, device='cuda')
            self.vlm(input_ids=torch.tensor([pending], device='cuda'), past_key_values=self.cache, use_cache=True, cache_position=cpos,
                     position_ids=(cpos + self.rd).view(1, 1, S).expand(3, 1, S), logits_to_keep=1)
        torch.cuda.synchronize(); t3 = time.perf_counter(); self.t = dict(prefill=t1 - t0, decode=t2 - t1, closing=t3 - t2, n=len(gen))
        Pf = P + len(closed); anchor = prompt_anchor + len(closed)
        cache = []
        for i in m.config.full_attention_layers:
            L = self.cache.layers[i]; cache.append((L.keys[:, :, :Pf].transpose(1, 2), L.values[:, :, :Pf].transpose(1, 2)))
        return cache, anchor, text, len(gen)
