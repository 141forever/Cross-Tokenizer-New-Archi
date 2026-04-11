"""
Cross-Attention Knowledge Distillation from Teacher KV Cache to Student Model.

Architecture:
  - Student model (decoder-only) is loaded and used to initialize a new model.
  - The new model inserts a CrossAttention layer between each SelfAttention and FFN.
  - Teacher model provides KV cache per alignment group.
  - GOLD algorithm aligns student/teacher token groups.
  - Cross attention: s_groups[i] attends to t_groups[0:i] KV cache to produce s_groups[i+1].
  - flash_attn_varlen_func is used for efficient variable-length cross attention.

Usage:
    python cross_attn_distill.py \
        --student_model "meta-llama/Llama-3.2-1B" \
        --teacher_model "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset_name "trl-lib/Capybara" \
        --output_dir "./cross_attn_distill_output" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-5
"""

import os
import copy
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logging.warning("flash_attn not available, falling back to PyTorch SDPA.")

from datasets import load_dataset
from accelerate import Accelerator

logger = logging.getLogger(__name__)


# ============================================================================
# 1. GOLD Alignment Groups (adapted from trl.experimental.gold)
# ============================================================================

def _build_alignment_groups_from_ids(
    student_ids: List[int],
    teacher_ids: List[int],
    student_tokenizer,
    teacher_tokenizer,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build alignment groups from student and teacher token IDs.
    
    Incrementally decode student and teacher tokens; whenever both sides
    produce the same visible text prefix, cut a group boundary. This ensures
    s_groups[i] and t_groups[i] decode to the same string.
    
    Returns:
        s_groups: list of token-index lists for the student
        t_groups: list of token-index lists for the teacher
    """
    s_groups: List[List[int]] = []
    t_groups: List[List[int]] = []

    s_idx = 0
    t_idx = 0
    s_buf: List[int] = []
    t_buf: List[int] = []

    while s_idx < len(student_ids) and t_idx < len(teacher_ids):
        # Extend student buffer one token at a time
        s_buf.append(student_ids[s_idx])
        s_idx += 1
        s_text = student_tokenizer.decode(s_buf, skip_special_tokens=True)

        # Extend teacher buffer until its decoded text covers the student text
        while t_idx < len(teacher_ids):
            t_buf.append(teacher_ids[t_idx])
            t_idx += 1
            t_text = teacher_tokenizer.decode(t_buf, skip_special_tokens=True)
            if len(t_text) >= len(s_text):
                break

        t_text = teacher_tokenizer.decode(t_buf, skip_special_tokens=True)

        # Check if texts match — if so, cut a group boundary
        if s_text == t_text:
            s_groups.append(list(s_buf))
            t_groups.append(list(t_buf))
            s_buf = []
            t_buf = []
        else:
            # Texts diverge: keep extending student until they align again
            # Try extending student side
            continue

    # Handle remaining tokens
    if s_buf or t_buf:
        # Extend both sides to consume remaining tokens
        while s_idx < len(student_ids):
            s_buf.append(student_ids[s_idx])
            s_idx += 1
        while t_idx < len(teacher_ids):
            t_buf.append(teacher_ids[t_idx])
            t_idx += 1
        if s_buf and t_buf:
            s_groups.append(s_buf)
            t_groups.append(t_buf)

    return s_groups, t_groups


def build_alignment_groups_batch(
    student_input_ids: torch.Tensor,       # (batch, seq_len_s)
    teacher_input_ids: torch.Tensor,       # (batch, seq_len_t)
    student_tokenizer,
    teacher_tokenizer,
) -> List[Tuple[List[List[int]], List[List[int]]]]:
    """Build alignment groups for each item in the batch."""
    batch_groups = []
    for b in range(student_input_ids.size(0)):
        s_ids = student_input_ids[b].tolist()
        t_ids = teacher_input_ids[b].tolist()
        # Remove padding
        s_ids = [x for x in s_ids if x != student_tokenizer.pad_token_id]
        t_ids = [x for x in t_ids if x != teacher_tokenizer.pad_token_id]
        s_groups, t_groups = _build_alignment_groups_from_ids(
            s_ids, t_ids, student_tokenizer, teacher_tokenizer
        )
        batch_groups.append((s_groups, t_groups))
    return batch_groups


# ============================================================================
# 2. Cross Attention Module
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention layer where Q comes from student hidden states and
    K, V come from teacher KV cache.
    
    Uses flash_attn_varlen_func for efficient variable-length cross attention
    when available.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, num_kv_heads: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        
        # Q projection from student hidden states
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        # K, V projections from teacher hidden states (will project from teacher dim)
        # These are initialized separately since teacher dim may differ
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Layer norm for the cross attention input
        self.layer_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        
        self.scale = head_dim ** -0.5

    def _flash_cross_attn_varlen(
        self,
        query: torch.Tensor,         # (total_q, num_heads, head_dim)
        key: torch.Tensor,           # (total_kv, num_kv_heads, head_dim)
        value: torch.Tensor,         # (total_kv, num_kv_heads, head_dim)
        cu_seqlens_q: torch.Tensor,  # (batch+1,) int32
        cu_seqlens_k: torch.Tensor,  # (batch+1,) int32
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Use flash_attn_varlen_func for cross attention (non-causal)."""
        out = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,  # cross attention is non-causal
        )
        return out

    def _sdpa_cross_attn(
        self,
        query: torch.Tensor,  # (batch, q_len, num_heads, head_dim)
        key: torch.Tensor,    # (batch, kv_len, num_kv_heads, head_dim)
        value: torch.Tensor,  # (batch, kv_len, num_kv_heads, head_dim)
    ) -> torch.Tensor:
        """Fallback: PyTorch SDPA for cross attention."""
        # Transpose to (batch, heads, seq, dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        
        # Handle GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return out.transpose(1, 2)  # (batch, q_len, heads, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,       # student hidden: (batch, seq_q, hidden_size)
        teacher_key: torch.Tensor,         # (batch, seq_kv, num_kv_heads, head_dim) or flat
        teacher_value: torch.Tensor,       # (batch, seq_kv, num_kv_heads, head_dim) or flat
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        use_varlen: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: student hidden states
            teacher_key, teacher_value: pre-projected K, V from teacher
            cu_seqlens_*: cumulative sequence lengths for varlen mode
            use_varlen: whether to use flash_attn_varlen_func
        """
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        bsz = hidden_states.size(0) if not use_varlen else None
        
        # Project Q from student hidden states
        if use_varlen:
            # hidden_states: (total_q, hidden_size)
            q = self.q_proj(hidden_states)
            q = q.view(-1, self.num_heads, self.head_dim)
        else:
            seq_q = hidden_states.size(1)
            q = self.q_proj(hidden_states)
            q = q.view(bsz, seq_q, self.num_heads, self.head_dim)
        
        # K, V are already provided from teacher
        if use_varlen and FLASH_ATTN_AVAILABLE:
            out = self._flash_cross_attn_varlen(
                q, teacher_key, teacher_value,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
            )
            out = out.view(-1, self.num_heads * self.head_dim)
            out = self.o_proj(out)
        else:
            out = self._sdpa_cross_attn(q, teacher_key, teacher_value)
            out = out.reshape(bsz, -1, self.num_heads * self.head_dim)
            out = self.o_proj(out)
        
        # Residual connection
        return residual + out


# ============================================================================
# 3. Modified Decoder Layer with Cross Attention
# ============================================================================

class DecoderLayerWithCrossAttention(nn.Module):
    """
    A decoder layer that mirrors the student's original layer but inserts
    a CrossAttention block between self-attention and FFN.
    
    Flow: input -> self_attn -> cross_attn (new) -> ffn -> output
    """

    def __init__(self, original_layer: nn.Module, config):
        super().__init__()
        # Copy the original self-attention and FFN sub-modules
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        
        # Copy layer norms — handle different naming conventions
        if hasattr(original_layer, 'input_layernorm'):
            self.input_layernorm = original_layer.input_layernorm
        if hasattr(original_layer, 'post_attention_layernorm'):
            self.post_attention_layernorm = original_layer.post_attention_layernorm
        
        # Determine dimensions from config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        
        # Create cross attention (newly initialized)
        self.cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )
        
        # Additional layer norm before FFN after cross attention
        self.post_cross_attn_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # Cross attention args
        teacher_key: Optional[torch.Tensor] = None,
        teacher_value: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        use_varlen: bool = False,
        **kwargs,
    ):
        # 1. Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        self_attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        if isinstance(self_attn_out, tuple):
            hidden_states = self_attn_out[0]
        else:
            hidden_states = self_attn_out
        hidden_states = residual + hidden_states
        
        # 2. Cross Attention (with teacher KV)
        if teacher_key is not None and teacher_value is not None:
            hidden_states = self.cross_attn(
                hidden_states=hidden_states,
                teacher_key=teacher_key,
                teacher_value=teacher_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                use_varlen=use_varlen,
            )
        
        # 3. FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if isinstance(self_attn_out, tuple) and len(self_attn_out) > 1:
            return (hidden_states,) + self_attn_out[1:]
        return (hidden_states,)


# ============================================================================
# 4. Full Model with Cross Attention
# ============================================================================

class StudentModelWithCrossAttention(nn.Module):
    """
    Wraps a decoder-only student model and inserts cross attention layers.
    All weights are initialized from the student model except cross attention.
    """

    def __init__(self, student_model: PreTrainedModel, config):
        super().__init__()
        self.config = config
        
        # Get the model's core transformer (handle different HF conventions)
        if hasattr(student_model, 'model'):
            base_model = student_model.model
        else:
            base_model = student_model
        
        # Copy embedding layers
        self.embed_tokens = base_model.embed_tokens
        
        # Copy and wrap each decoder layer
        self.layers = nn.ModuleList()
        original_layers = base_model.layers
        for layer in original_layers:
            wrapped = DecoderLayerWithCrossAttention(layer, config)
            self.layers.append(wrapped)
        
        # Copy final norm
        self.norm = base_model.norm
        
        # Copy lm_head
        if hasattr(student_model, 'lm_head'):
            self.lm_head = student_model.lm_head
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        # Per-layer teacher KV (list of (key, value) tuples, one per layer)
        teacher_kv_per_layer: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        use_varlen: bool = False,
    ) -> CausalLMOutputWithPast:
        
        hidden_states = self.embed_tokens(input_ids)
        
        for i, layer in enumerate(self.layers):
            t_key, t_value = None, None
            if teacher_kv_per_layer is not None and i < len(teacher_kv_per_layer):
                t_key, t_value = teacher_kv_per_layer[i]
            
            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                teacher_key=t_key,
                teacher_value=t_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                use_varlen=use_varlen,
            )
            hidden_states = layer_out[0]
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )


# ============================================================================
# 5. Teacher KV Cache Extractor
# ============================================================================

class TeacherKVExtractor:
    """
    Extracts KV cache from the teacher model for each layer, 
    organized by alignment groups.
    """

    def __init__(self, teacher_model: PreTrainedModel, teacher_tokenizer):
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_full_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run teacher forward pass and extract KV cache from all layers.
        
        Returns:
            List of (key, value) per layer, each shaped 
            (batch, num_kv_heads, seq_len, head_dim)
        """
        outputs = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )
        
        # past_key_values: tuple of (key, value) per layer
        # Each key/value: (batch, num_kv_heads, seq_len, head_dim)
        kv_cache = []
        for layer_kv in outputs.past_key_values:
            k, v = layer_kv[0], layer_kv[1]
            kv_cache.append((k, v))
        
        return kv_cache

    @torch.no_grad()
    def extract_grouped_kv(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        t_groups_batch: Optional[List[List[List[int]]]] = None,
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Extract KV cache and split by alignment groups.
        
        Returns:
            per_layer_grouped_kv: List[num_layers] of List[num_groups] of (K, V)
        """
        full_kv = self.get_full_kv_cache(input_ids, attention_mask)
        
        if t_groups_batch is None:
            return full_kv
        
        num_layers = len(full_kv)
        # For simplicity, process first sample in batch
        # (extend to full batch as needed)
        
        per_layer_grouped = []
        for layer_idx in range(num_layers):
            k, v = full_kv[layer_idx]  # (batch, heads, seq, dim)
            grouped_kv = []
            
            for b in range(k.size(0)):
                t_groups = t_groups_batch[b]
                pos = 0
                batch_grouped = []
                for group_ids in t_groups:
                    group_len = len(group_ids)
                    k_group = k[b:b+1, :, pos:pos+group_len, :]
                    v_group = v[b:b+1, :, pos:pos+group_len, :]
                    batch_grouped.append((k_group, v_group))
                    pos += group_len
                grouped_kv.append(batch_grouped)
            
            per_layer_grouped.append(grouped_kv)
        
        return per_layer_grouped


# ============================================================================
# 6. Cross Attention with Cumulative Teacher KV (core logic)
# ============================================================================

def prepare_cross_attn_inputs_for_group(
    student_hidden: torch.Tensor,    # (batch, total_s_len, hidden)
    s_groups: List[List[List[int]]],  # per-batch student groups
    teacher_kv_grouped: List[List[Tuple[torch.Tensor, torch.Tensor]]],  # per-batch grouped KV
    group_idx: int,                   # which group we're generating
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Prepare cross-attention inputs for a specific alignment group.
    
    For generating s_groups[group_idx+1], the cross attention attends to
    t_groups[0:group_idx+1] KV cache with q from s_groups[0:group_idx+1].
    
    Uses flash_attn_varlen format: flattened (total_q, heads, dim) with cu_seqlens.
    """
    batch_size = len(s_groups)
    
    q_list = []
    k_list = []
    v_list = []
    q_seqlens = []
    k_seqlens = []
    
    for b in range(batch_size):
        # Q: student tokens from groups [0 .. group_idx]
        s_start = sum(len(g) for g in s_groups[b][:group_idx])
        s_end = sum(len(g) for g in s_groups[b][:group_idx + 1])
        q_tokens = student_hidden[b, s_start:s_end, :]  # (q_len, hidden)
        q_list.append(q_tokens)
        q_seqlens.append(s_end - s_start)
        
        # K, V: teacher KV from groups [0 .. group_idx]
        k_parts = []
        v_parts = []
        for g in range(group_idx + 1):
            if g < len(teacher_kv_grouped[b]):
                kg, vg = teacher_kv_grouped[b][g]
                # kg: (1, heads, group_len, dim) -> (group_len, heads, dim)
                k_parts.append(kg.squeeze(0).transpose(0, 1))
                v_parts.append(vg.squeeze(0).transpose(0, 1))
        
        if k_parts:
            k_cat = torch.cat(k_parts, dim=1)  # (heads, total_kv, dim)
            v_cat = torch.cat(v_parts, dim=1)
            k_list.append(k_cat.transpose(0, 1))  # (total_kv, heads, dim)
            v_list.append(v_cat.transpose(0, 1))
            k_seqlens.append(k_cat.size(1))
        else:
            k_seqlens.append(0)
    
    # Build cu_seqlens
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_seqlens[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + k_seqlens[i]
    
    # Concatenate across batch
    if q_list:
        q_flat = torch.cat(q_list, dim=0)  # (total_q, hidden)
    else:
        q_flat = torch.empty(0, student_hidden.size(-1), device=device)
    
    if k_list:
        k_flat = torch.cat(k_list, dim=0)  # (total_kv, heads, dim)
        v_flat = torch.cat(v_list, dim=0)
    else:
        k_flat = torch.empty(0, num_heads, head_dim, device=device)
        v_flat = torch.empty(0, num_heads, head_dim, device=device)
    
    max_q = max(q_seqlens) if q_seqlens else 0
    max_k = max(k_seqlens) if k_seqlens else 0
    
    return {
        "q_hidden": q_flat,
        "k_flat": k_flat,
        "v_flat": v_flat,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_q,
        "max_seqlen_k": max_k,
        "q_seqlens": q_seqlens,
    }


# ============================================================================
# 7. Training Dataset
# ============================================================================

class CrossAttnDistillDataset(Dataset):
    """Dataset that prepares paired student/teacher inputs."""
    
    def __init__(
        self,
        dataset,
        student_tokenizer,
        teacher_tokenizer,
        max_length: int = 512,
    ):
        self.dataset = dataset
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if 'messages' in item:
            # ChatML format
            text = self.student_tokenizer.apply_chat_template(
                item['messages'], tokenize=False
            )
        elif 'text' in item:
            text = item['text']
        elif 'content' in item:
            text = item['content']
        else:
            text = str(list(item.values())[0])
        
        # Tokenize for student
        s_enc = self.student_tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt',
        )
        # Tokenize for teacher
        t_enc = self.teacher_tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt',
        )
        
        return {
            'student_input_ids': s_enc['input_ids'].squeeze(0),
            'student_attention_mask': s_enc['attention_mask'].squeeze(0),
            'teacher_input_ids': t_enc['input_ids'].squeeze(0),
            'teacher_attention_mask': t_enc['attention_mask'].squeeze(0),
            'text': text,
        }


def collate_fn(batch):
    return {
        'student_input_ids': torch.stack([b['student_input_ids'] for b in batch]),
        'student_attention_mask': torch.stack([b['student_attention_mask'] for b in batch]),
        'teacher_input_ids': torch.stack([b['teacher_input_ids'] for b in batch]),
        'teacher_attention_mask': torch.stack([b['teacher_attention_mask'] for b in batch]),
    }


# ============================================================================
# 8. Grouped Forward Pass (core training step)
# ============================================================================

def grouped_forward_step(
    model: StudentModelWithCrossAttention,
    teacher_kv_extractor: TeacherKVExtractor,
    student_input_ids: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    student_tokenizer,
    teacher_tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Perform the grouped forward pass:
    1. Build alignment groups via GOLD algorithm.
    2. Get teacher KV cache grouped by alignment.
    3. For each group i, cross-attend s_groups[0:i] with t_groups[0:i]
       to help generate s_groups[i+1].
    4. Compute language modeling loss.
    """
    batch_size = student_input_ids.size(0)
    
    # Step 1: Build alignment groups
    batch_groups = build_alignment_groups_batch(
        student_input_ids, teacher_input_ids,
        student_tokenizer, teacher_tokenizer,
    )
    s_groups_batch = [bg[0] for bg in batch_groups]
    t_groups_batch = [bg[1] for bg in batch_groups]
    
    # Step 2: Get teacher KV cache (full forward, then slice by groups)
    teacher_full_kv = teacher_kv_extractor.get_full_kv_cache(
        teacher_input_ids, teacher_attention_mask
    )
    
    # Project teacher KV into student's cross-attn dimension space
    # For simplicity, we use the raw teacher KV and let cross_attn.k_proj/v_proj handle it
    # But since teacher hidden dim may differ, we need to handle this
    
    num_teacher_layers = len(teacher_full_kv)
    num_student_layers = len(model.layers)
    
    # Map teacher layers to student layers (linearly)
    layer_mapping = []
    for s_layer in range(num_student_layers):
        t_layer = int(s_layer * num_teacher_layers / num_student_layers)
        t_layer = min(t_layer, num_teacher_layers - 1)
        layer_mapping.append(t_layer)
    
    # Slice teacher KV by groups for each layer
    per_layer_grouped_kv = []
    for s_layer in range(num_student_layers):
        t_layer = layer_mapping[s_layer]
        k_full, v_full = teacher_full_kv[t_layer]  # (batch, heads, seq, dim)
        
        batch_grouped = []
        for b in range(batch_size):
            t_groups = t_groups_batch[b]
            pos = 0
            sample_grouped = []
            for group_ids in t_groups:
                group_len = len(group_ids)
                k_g = k_full[b:b+1, :, pos:pos+group_len, :]
                v_g = v_full[b:b+1, :, pos:pos+group_len, :]
                sample_grouped.append((k_g, v_g))
                pos += group_len
            batch_grouped.append(sample_grouped)
        per_layer_grouped_kv.append(batch_grouped)
    
    # Step 3: Forward pass with group-by-group cross attention
    # We do a full forward pass through the student model.
    # At each layer, we build cumulative teacher KV for cross attention.
    
    # Compute number of groups (minimum across batch)
    num_groups = min(len(sg) for sg in s_groups_batch)
    
    if num_groups <= 1:
        # Fallback: just do a normal forward pass without cross attention
        labels = student_input_ids.clone()
        labels[student_attention_mask == 0] = -100
        outputs = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=labels,
        )
        return outputs.loss
    
    # Build cumulative KV for each layer: for group i, accumulate groups [0..i-1]
    # Then do a single forward pass with the accumulated KV
    
    # For efficiency, we do ONE forward pass through the model, but provide
    # the accumulated teacher KV at each layer.
    # The cross attention at each layer sees ALL teacher groups up to the 
    # maximum alignment position.
    
    # Build per-layer accumulated KV
    teacher_kv_per_layer = []
    for s_layer in range(num_student_layers):
        # Accumulate all teacher groups for this layer
        k_parts_batch = []
        v_parts_batch = []
        max_kv_len = 0
        
        for b in range(batch_size):
            k_parts = []
            v_parts = []
            # Use all groups except the last (teacher provides context for prediction)
            for g in range(min(num_groups - 1, len(per_layer_grouped_kv[s_layer][b]))):
                kg, vg = per_layer_grouped_kv[s_layer][b][g]
                k_parts.append(kg)  # (1, heads, group_len, dim)
                v_parts.append(vg)
            
            if k_parts:
                k_cat = torch.cat(k_parts, dim=2)  # (1, heads, total_kv, dim)
                v_cat = torch.cat(v_parts, dim=2)
                k_parts_batch.append(k_cat)
                v_parts_batch.append(v_cat)
                max_kv_len = max(max_kv_len, k_cat.size(2))
            else:
                k_parts_batch.append(None)
                v_parts_batch.append(None)
        
        # Pad to same length and stack
        if max_kv_len > 0 and all(x is not None for x in k_parts_batch):
            padded_k = []
            padded_v = []
            for b in range(batch_size):
                k = k_parts_batch[b]
                v = v_parts_batch[b]
                pad_len = max_kv_len - k.size(2)
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                padded_k.append(k)
                padded_v.append(v)
            
            # (batch, heads, kv_len, dim) -> (batch, kv_len, heads, dim) for cross attn
            k_stacked = torch.cat(padded_k, dim=0).transpose(1, 2)
            v_stacked = torch.cat(padded_v, dim=0).transpose(1, 2)
            teacher_kv_per_layer.append((k_stacked, v_stacked))
        else:
            teacher_kv_per_layer.append((None, None))
    
    # Filter out None pairs
    valid_kv = []
    for k, v in teacher_kv_per_layer:
        if k is not None:
            valid_kv.append((k, v))
        else:
            # Create zero-size placeholder to skip cross attn
            valid_kv.append((None, None))
    
    # Forward pass
    labels = student_input_ids.clone()
    labels[student_attention_mask == 0] = -100
    
    outputs = model(
        input_ids=student_input_ids,
        attention_mask=student_attention_mask,
        labels=labels,
        teacher_kv_per_layer=valid_kv,
    )
    
    return outputs.loss


# ============================================================================
# 9. Training Loop
# ============================================================================

@dataclass
class DistillArgs:
    student_model: str = "meta-llama/Llama-3.2-1B"
    teacher_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_name: str = "trl-lib/Capybara"
    dataset_split: str = "train"
    max_samples: int = -1
    output_dir: str = "./cross_attn_distill_output"
    max_length: int = 512
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42


def train(args: DistillArgs):
    """Main training function."""
    
    # Set up accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else ("fp16" if args.fp16 else "no"),
    )
    
    logger.info(f"Loading student model: {args.student_model}")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation="flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa",
    )
    student_config = student_model.config
    
    logger.info(f"Loading teacher model: {args.teacher_model}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation="flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa",
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    # Build the student model with cross attention
    logger.info("Building student model with cross attention layers...")
    model = StudentModelWithCrossAttention(student_model, student_config)
    
    # Enable gradient checkpointing on original layers
    if args.gradient_checkpointing:
        for layer in model.layers:
            if hasattr(layer.self_attn, 'gradient_checkpointing'):
                layer.self_attn.gradient_checkpointing = True
    
    # Free original student model to save memory
    del student_model
    torch.cuda.empty_cache()
    
    # Initialize teacher KV extractor
    teacher_kv_extractor = TeacherKVExtractor(teacher_model, teacher_tokenizer)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    raw_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_samples > 0:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))
    
    train_dataset = CrossAttnDistillDataset(
        raw_dataset, student_tokenizer, teacher_tokenizer, args.max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer
    # Only train cross attention parameters + student model params
    cross_attn_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'cross_attn' in name:
            cross_attn_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': cross_attn_params, 'lr': args.learning_rate},
        {'params': other_params, 'lr': args.learning_rate * 0.1},  # lower LR for pretrained params
    ], weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    teacher_model = teacher_model.to(accelerator.device)
    teacher_kv_extractor.teacher_model = teacher_model
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = grouped_forward_step(
                    model=accelerator.unwrap_model(model),
                    teacher_kv_extractor=teacher_kv_extractor,
                    student_input_ids=batch['student_input_ids'],
                    student_attention_mask=batch['student_attention_mask'],
                    teacher_input_ids=batch['teacher_input_ids'],
                    teacher_attention_mask=batch['teacher_attention_mask'],
                    student_tokenizer=student_tokenizer,
                    teacher_tokenizer=teacher_tokenizer,
                    device=accelerator.device,
                )
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                lr = optimizer.param_groups[0]['lr']
                accelerator.print(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                accelerator.print(f"Saved checkpoint to {save_path}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), os.path.join(final_path, "model.pt"))
        student_tokenizer.save_pretrained(final_path)
        accelerator.print(f"Saved final model to {final_path}")
    
    accelerator.print("Training completed!")


# ============================================================================
# 10. Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Attention Knowledge Distillation")
    parser.add_argument("--student_model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trl-lib/Capybara")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="./cross_attn_distill_output")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    cli_args = parser.parse_args()
    
    distill_args = DistillArgs(**vars(cli_args))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    torch.manual_seed(distill_args.seed)
    
    train(distill_args)