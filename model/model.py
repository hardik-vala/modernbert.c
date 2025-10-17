from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    # default hyperparameters for answerdotai/ModernBERT-base
    dim: int = 768
    vocab_size: int = 50368
    n_layers: int = 22
    norm_eps: float = 1e-5
    dropout: float = 0.0
    max_seq_len: int = 8192
    n_heads: int = 12
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    intermediate_dim: int = 1152  # MLP hidden layer size
    global_attn_every_n_layers: int = 3
    local_attention: int = 128  # local attention window size
    approximate_gelu: bool = True  # whether to use approximate gelu or exact gelu


class MLP(nn.Module):

    def __init__(self, args: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.args = args
        self.layer_id = layer_id  # for debugging forward
        self.approximate_gelu = args.approximate_gelu
        self.wi = nn.Linear(args.dim, args.intermediate_dim * 2, bias=False)
        self.act = F.gelu  # GELU activation
        self.dropout = nn.Dropout(args.dropout)
        self.wo = nn.Linear(args.intermediate_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input, gate = self.wi(x).chunk(2, dim=-1)
        act = self.act(input, approximate=("tanh" if self.approximate_gelu else "none"))
        return self.wo(self.dropout(act * gate))


class Attn(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.args = args
        self.layer_id = layer_id  # for debugging foward
        self.dim = args.dim
        self.n_heads = args.n_heads
        assert (
            self.dim % self.n_heads == 0
        ), "embedding dimension must be divisible by number of heads"
        self.head_dim = self.dim // self.n_heads

        self.dropout = args.dropout
        self.wqkv = nn.Linear(self.dim, 3 * self.dim, bias=False)

        if layer_id is not None and layer_id % args.global_attn_every_n_layers != 0:
            self.local_attention = (
                args.local_attention // 2,
                args.local_attention // 2,
            )
            self.rope_theta = args.local_rope_theta
        else:
            self.local_attention = (-1, -1)
            self.rope_theta = args.global_rope_theta

        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.out_drop = nn.Dropout(self.dropout)

    def _get_rope_freqs(
        self,
        position_ids: torch.LongTensor,  # (1, seq_len)
    ) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )  # (head_dim/2,)
        freqs = torch.outer(position_ids.squeeze(0).float(), inv_freq).unsqueeze(
            0
        )  # (1, seq_len, head_dim/2)
        freqs = torch.cat((freqs, freqs), dim=-1)  # (1, seq_len, head_dim)
        cos = freqs.cos()  # real part
        sin = freqs.sin()  # imaginary part

        return cos, sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_emb(
        self,
        q: torch.Tensor,  # (bsz, n_heads, seq_len, head_dim)
        k: torch.Tensor,  # (bsz, n_heads, seq_len, head_dim
        cos: torch.Tensor,  # (1, seq_len, head_dim)
        sin: torch.Tensor,  # (1, seq_len, head_dim)
    ) -> torch.Tensor:
        cos = cos.unsqueeze(1)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)  # (1, 1, seq_len, head_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        x: torch.Tensor,  # (bsz, seq_len, dim)
        position_ids: torch.LongTensor,  # (1, seq_len)
        output_attentions: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        xqkv = self.wqkv(x)  # (bsz, seq_len, 3 * dim)
        xqkv = xqkv.view(bsz, seq_len, 3, self.n_heads, self.head_dim)

        cos, sin = self._get_rope_freqs(position_ids=position_ids)

        xq, xk, xv = xqkv.transpose(3, 1).unbind(
            dim=2
        )  # each is (bsz, n_heads, seq_len, head_dim)
        xq, xk = self._apply_rotary_emb(
            xq, xk, cos=cos, sin=sin
        )  # each is (bsz, n_heads, seq_len, head_dim)

        scale = self.head_dim**-0.5
        attn_scores = (
            torch.matmul(xq, xk.transpose(-2, -1)) * scale
        )  # (bsz, n_heads, seq_len, seq_len)

        if self.local_attention != (-1, -1):
            # Create sliding window mask
            rows = torch.arange(seq_len, device=x.device).unsqueeze(0)
            distance = torch.abs(rows - rows.T)
            window_size = self.local_attention[0] + self.local_attention[1]
            window_mask = distance <= window_size // 2
            # Apply mask (set positions outside window to -inf before softmax)
            attn_scores = attn_scores.masked_fill(
                ~window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_scores = nn.functional.softmax(
            attn_scores, dim=-1
        )  # (bsz, n_heads, seq_len, seq_len)
        attn_scores = self.attn_dropout(attn_scores)
        attn_output = torch.matmul(attn_scores, xv)  # (bsz, n_heads, seq_len, head_dim)
        # restore time as batch dimension and concat heads
        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # (bsz, seq_len, n_heads, head_dim)
        attn_output = attn_output.view(bsz, -1, self.dim)  # (bsz, seq_len, dim)
        attn_output = self.out_drop(self.wo(attn_output))  # (bsz, seq_len, dim)

        if output_attentions:
            return (attn_output, attn_scores)
        return (attn_output,)


class TransformerLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.dim = args.dim
        assert (
            self.dim % args.n_heads == 0
        ), "embedding dimension must be divisible by number of heads"
        self.head_dim = args.dim // args.n_heads

        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(self.dim, eps=args.norm_eps, bias=False)

        self.attn = Attn(args, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(self.dim, eps=args.norm_eps, bias=False)
        self.mlp = MLP(args, layer_id=layer_id)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        attn_outputs = self.attn.forward(
            self.attn_norm(x),
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        h = x + attn_outputs[0]
        out = h + self.mlp(self.mlp_norm(h))
        return (out,) + attn_outputs[1:]


@dataclass
class ModelOutput:
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None

@dataclass
class ClassifierOutput(ModelOutput):
    logits: torch.FloatTensor = None

class ModernBERT(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers


class ModernBERTBase(ModernBERT):

    def __init__(self, args: ModelArgs):
        super().__init__(args)

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim)
        self.norm = nn.LayerNorm(self.dim, eps=args.norm_eps, bias=False)
        self.dropout = nn.Dropout(args.dropout, inplace=False)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerLayer(layer_id, args))
        self.final_norm = nn.LayerNorm(self.dim, eps=args.norm_eps, bias=False)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        h = self.norm(h)
        h = self.dropout(h)
        return h

    def forward(
        self,
        tokens: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        attentions = ()
        hidden_states = ()

        position_ids = torch.arange(tokens.size(1), dtype=torch.long).unsqueeze(
            0
        )  # (1, seq_len)

        h = self._embed(tokens)

        for layer in self.layers:
            if output_hidden_states:
                hidden_states = hidden_states + (h,)

            layer_outputs = layer(
                h, position_ids=position_ids, output_attentions=output_attentions
            )
            h = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                attentions = attentions + (layer_outputs[1],)
        if output_hidden_states:
            hidden_states = hidden_states + (h,)

        h = self.final_norm(h)

        return ModelOutput(
            last_hidden_state=h,
            hidden_states=hidden_states,
            attentions=attentions,
        )

class ModernBERTPredictionHead(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.dense = nn.Linear(args.dim, args.dim, bias=False)
        self.act = F.gelu  # GELU activation
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dense(x)
        h = self.act(h)
        h = self.norm(h)
        return h

class ModernBERTForTokenClassification(ModernBERT):

    def __init__(self, args: ModelArgs, num_labels: int):
        super().__init__(args)

        self.num_labels = num_labels
        self.model = ModernBERTBase(args)
        self.head = ModernBERTPredictionHead(args)
        self.dropout = nn.Dropout(args.dropout, inplace=False)
        self.classifier = nn.Linear(args.dim, num_labels, bias=True)

    def forward(
        self,
        tokens: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        outputs = self.model(
            tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        h = outputs.last_hidden_state
        h = self.head(h)
        h = self.dropout(h)
        logits = self.classifier(h)

        return ClassifierOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
