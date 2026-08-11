import inspect
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
@dataclass
class ModelArgs:
    """Architecture arguments for the Llama-style speech transformer."""

    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    prefix_vocab_size: Optional[int] = None
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    block_size: int = 2048
    dropout: float = 0.0
    prefix_pad: Optional[int] = None

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.block_size, args.block_size), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        cache_pos: int = 0,
        cache_size: int = 0,
    ):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if kv_cache is not None:
            if kv_cache[0] is None:
                kv_cache[0] = xk.new_empty(
                    bsz, cache_size, self.n_local_kv_heads, self.head_dim
                )
                kv_cache[1] = xv.new_empty(
                    bsz, cache_size, self.n_local_kv_heads, self.head_dim
                )
            cache_end = cache_pos + seqlen
            kv_cache[0][:, cache_pos:cache_end].copy_(xk)
            kv_cache[1][:, cache_pos:cache_end].copy_(xv)
            xk = kv_cache[0][:, :cache_end]
            xv = kv_cache[1][:, :cache_end]

        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=attention_mask is None and cache_pos == 0,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is None and cache_pos == 0:
                assert hasattr(self, 'mask')
                scores = scores + self.mask[:, :, :seqlen, :seqlen]
            elif attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x,
        freqs_cos,
        freqs_sin,
        attention_mask=None,
        kv_cache=None,
        cache_pos=0,
        cache_size=0,
    ):
        h = x + self.attention.forward(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            attention_mask,
            kv_cache,
            cache_pos,
            cache_size,
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class TransformerLM(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        if params.dim % params.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if params.n_kv_heads is not None and params.n_heads % params.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.params = params
        self.vocab_size = params.vocab_size
        self.prefix_vocab_size = params.prefix_vocab_size
        self.prefix_pad = params.prefix_pad
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        if self.prefix_vocab_size is not None:
            self.prefix_embeddings = nn.Embedding(
                params.prefix_vocab_size + (0 if self.prefix_pad is None else 1), params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.block_size)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        prefixes: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("tokens must have shape [batch, non-empty sequence]")
        if targets is not None and mask is not None and mask.shape != targets.shape:
            raise ValueError("mask and targets must have the same shape")
        if self.prefix_vocab_size is None and prefixes is not None:
            raise ValueError("this model was not configured for prefix tokens")
        if self.prefix_vocab_size is not None and prefixes is None:
            raise ValueError("this model requires prefix tokens")

        h = self.tok_embeddings(tokens)
        prefix_end = 0
        predictors_mask = None
        attention_mask = None
        if self.prefix_vocab_size is not None:
            if prefixes.ndim != 2 or prefixes.size(0) != tokens.size(0) or prefixes.size(1) == 0:
                raise ValueError("prefixes must have shape [batch, non-empty sequence]")
            prefix_valid = torch.ones_like(prefixes, dtype=torch.bool)
            aligned_prefixes = prefixes
            if self.prefix_pad is not None:
                source_valid = prefixes != self.prefix_pad
                torch._assert(
                    torch.all(source_valid.sum(1) > 0),
                    "each padded prefix must contain at least one token",
                )
                torch._assert(
                    ~torch.any(source_valid & ((~source_valid).cumsum(1) > 0)),
                    "prefix padding must be a contiguous suffix",
                )
                prefix_width = prefixes.size(1)
                lengths = source_valid.sum(1)
                positions = torch.arange(prefix_width, device=prefixes.device).unsqueeze(0)
                shifts = prefix_width - lengths
                source_positions = positions - shifts.unsqueeze(1)
                prefix_valid = source_positions >= 0
                gathered = prefixes.gather(1, source_positions.clamp_min(0))
                aligned_prefixes = torch.where(
                    prefix_valid,
                    gathered,
                    torch.full_like(prefixes, self.prefix_pad),
                )

            prefix_emb = self.prefix_embeddings(aligned_prefixes)
            h = torch.concat((prefix_emb, h), dim=1)
            if self.prefix_pad is not None:
                valid_positions = torch.cat(
                    (prefix_valid, torch.ones_like(tokens, dtype=torch.bool)), dim=1
                )
                seq_len = valid_positions.size(1)
                causal = torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=tokens.device
                ).tril()
                attention_mask = causal.view(1, 1, seq_len, seq_len) & valid_positions[:, None, None, :]
                identity = torch.eye(seq_len, dtype=torch.bool, device=tokens.device)
                attention_mask = attention_mask | (
                    (~valid_positions)[:, None, :, None]
                    & identity.view(1, 1, seq_len, seq_len)
                )
                prefix_mask = torch.zeros_like(prefixes, dtype=torch.bool)
                prefix_mask[:, -1] = True
                predictors_mask = torch.concatenate((prefix_mask, torch.ones_like(tokens, dtype=bool)), dim=1)
            else:
                prefix_end = prefixes.size(1) - 1
        seqlen = h.size(1)
        if seqlen > self.params.block_size:
            raise ValueError(f"sequence length {seqlen} exceeds block_size {self.params.block_size}")
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, attention_mask)
        h = self.norm(h)

        if targets is not None:
            if predictors_mask is not None:
                h = h[predictors_mask]
            else:
                h = h[:, prefix_end:]
            expected_targets = tokens.size(1) + int(self.prefix_vocab_size is not None)
            if targets.shape != (tokens.size(0), expected_targets):
                raise ValueError(
                    f"targets must have shape {(tokens.size(0), expected_targets)}"
                )
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction="mean" if mask is None else "none",
            )
            if mask is not None:
                self.last_loss = self.last_loss.view_as(targets)
                self.last_loss *= mask
                self.last_loss = self.last_loss.sum(1) / mask.sum(1).clamp_min(1)
        else:
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def _cached_forward(
        self,
        tokens: torch.Tensor,
        kv_cache: list[list],
        cache_valid: torch.Tensor,
        cache_pos: int,
        prefixes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        if cache_pos == 0:
            h = self.tok_embeddings(tokens)
            valid = torch.ones_like(tokens, dtype=torch.bool)
            attention_mask = None
            if prefixes is not None:
                prefix_valid = torch.ones_like(prefixes, dtype=torch.bool)
                aligned_prefixes = prefixes
                if self.prefix_pad is not None:
                    source_valid = prefixes != self.prefix_pad
                    if not bool(torch.all(source_valid.sum(1) > 0)):
                        raise ValueError("each padded prefix must contain at least one token")
                    if bool(torch.any(source_valid & ((~source_valid).cumsum(1) > 0))):
                        raise ValueError("prefix padding must be a contiguous suffix")
                    lengths = source_valid.sum(1)
                    width = prefixes.size(1)
                    positions = torch.arange(width, device=prefixes.device).unsqueeze(0)
                    source_positions = positions - (width - lengths).unsqueeze(1)
                    prefix_valid = source_positions >= 0
                    aligned_prefixes = prefixes.gather(1, source_positions.clamp_min(0))
                    aligned_prefixes = torch.where(
                        prefix_valid,
                        aligned_prefixes,
                        torch.full_like(aligned_prefixes, self.prefix_pad),
                    )
                h = torch.cat((self.prefix_embeddings(aligned_prefixes), h), dim=1)
                valid = torch.cat((prefix_valid, valid), dim=1)
                if self.prefix_pad is not None:
                    length = valid.size(1)
                    causal = torch.ones((length, length), dtype=torch.bool, device=h.device).tril()
                    attention_mask = causal.view(1, 1, length, length) & valid[:, None, None, :]
                    identity = torch.eye(length, dtype=torch.bool, device=h.device)
                    attention_mask |= (
                        (~valid)[:, None, :, None]
                        & identity.view(1, 1, length, length)
                    )
        else:
            if prefixes is not None:
                raise ValueError("prefixes are only accepted during cache prefill")
            h = self.tok_embeddings(tokens)
            valid = torch.ones_like(tokens, dtype=torch.bool)
            attention_mask = None

        length = h.size(1)
        cache_end = cache_pos + length
        cache_valid[:, cache_pos:cache_end] = valid
        if cache_pos and self.prefix_pad is not None:
            attention_mask = cache_valid[:, None, None, :cache_end]

        h = self.dropout(h)
        freqs_cos = self.freqs_cos[cache_pos:cache_end]
        freqs_sin = self.freqs_sin[cache_pos:cache_end]
        for layer, layer_cache in zip(self.layers, kv_cache):
            h = layer(
                h,
                freqs_cos,
                freqs_sin,
                attention_mask,
                layer_cache,
                cache_pos,
                self.params.block_size,
            )
        logits = self.output(self.norm(h[:, [-1], :]))
        return logits, cache_end

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def num_parameters(self) -> int:
        """Return the number of unique trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, prefixes=None, temperature=1.0, top_k=None):
        """Complete a token prompt using a preallocated per-layer KV cache."""
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("prompt must have shape [batch, non-empty sequence]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        if self.prefix_vocab_size is None and prefixes is not None:
            raise ValueError("this model was not configured for prefix tokens")
        if self.prefix_vocab_size is not None and prefixes is None:
            raise ValueError("this model requires prefix tokens")
        if prefixes is not None and (
            prefixes.ndim != 2
            or prefixes.size(0) != idx.size(0)
            or prefixes.size(1) == 0
        ):
            raise ValueError("prefixes must have shape [batch, non-empty sequence]")
        prefix_length = 0 if prefixes is None else prefixes.size(1)
        if prefix_length + idx.size(1) + max_new_tokens > self.params.block_size:
            raise ValueError("prefix, prompt, and continuation exceed block_size")
        if max_new_tokens == 0:
            return idx

        kv_cache = [[None, None] for _ in self.layers]
        cache_valid = torch.zeros(
            idx.size(0), self.params.block_size, dtype=torch.bool, device=idx.device
        )
        logits, cache_pos = self._cached_forward(
            idx, kv_cache, cache_valid, 0, prefixes
        )
        for step in range(max_new_tokens):
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if step + 1 < max_new_tokens:
                logits, cache_pos = self._cached_forward(
                    idx_next, kv_cache, cache_valid, cache_pos
                )

        return idx

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path, *, device: str | torch.device = "cpu"):
        """Load a native SLM directory containing config.json and safetensors."""
        root = Path(path_or_repo)
        if not root.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("Install huggingface-hub to load a remote model") from exc
            root = Path(snapshot_download(str(path_or_repo)))

        config = json.loads((root / "config.json").read_text())
        if config.get("format_version") != 1:
            raise ValueError(f"unsupported checkpoint format: {config.get('format_version')}")
        allowed = {f.name for f in fields(ModelArgs)}
        unknown = set(config["model_args"]) - allowed
        if unknown:
            raise ValueError(f"unknown model arguments: {sorted(unknown)}")
        args = ModelArgs(**config["model_args"])
        model = cls(args)
        state = _load_safetensors(root)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {"output.weight"} if "tok_embeddings.weight" in state else set()
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(f"incompatible weights: missing={missing}, unexpected={unexpected}")
        model.to(device)
        model.eval()
        return model

    def save_pretrained(self, output_dir: str | Path, *, max_shard_size: int = 4_000_000_000):
        """Write model-only FP32 safetensors and a portable native config."""
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("Install safetensors to export a model") from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if max_shard_size < 1:
            raise ValueError("max_shard_size must be positive")
        if list(output_dir.glob("model*.safetensors*")):
            raise FileExistsError(f"model weights already exist in {output_dir}")
        config = {
            "format_version": 1,
            "model_type": "slm",
            "architectures": [self.__class__.__name__],
            "model_args": asdict(self.params),
            "torch_dtype": "float32",
            "tied_weights": ["tok_embeddings.weight", "output.weight"],
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        state = {}
        seen = set()
        for name, tensor in self.state_dict().items():
            storage = tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
            key = (storage.data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
            if key in seen:
                continue
            seen.add(key)
            state[name] = tensor.detach().float().cpu().contiguous()

        shards = []
        current, current_size = {}, 0
        for name, tensor in state.items():
            size = tensor.numel() * tensor.element_size()
            if current and current_size + size > max_shard_size:
                shards.append(current)
                current, current_size = {}, 0
            current[name] = tensor
            current_size += size
        if current:
            shards.append(current)

        if len(shards) == 1:
            save_file(shards[0], output_dir / "model.safetensors", metadata={"format": "pt"})
            return

        weight_map = {}
        for index, shard in enumerate(shards, 1):
            name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, output_dir / name, metadata={"format": "pt"})
            weight_map.update({key: name for key in shard})
        index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())}, "weight_map": weight_map}
        (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def _load_safetensors(root: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Install safetensors to load this model") from exc
    single = root / "model.safetensors"
    index_path = root / "model.safetensors.index.json"
    if single.exists() and index_path.exists():
        raise ValueError(f"ambiguous single-file and sharded weights in {root}")
    if single.exists():
        return load_file(single, device="cpu")
    if not index_path.exists():
        raise FileNotFoundError(f"no safetensors weights found in {root}")
    index = json.loads(index_path.read_text())
    if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError("invalid safetensors index")
    state = {}
    for filename in sorted(set(index["weight_map"].values())):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"invalid safetensors shard name: {filename}")
        state.update(load_file(root / filename, device="cpu"))
    return state
