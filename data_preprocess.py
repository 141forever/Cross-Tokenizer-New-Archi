import os
import math
import logging
import json
import pdb
import time
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Set ddp
# ============================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main(rank):
    return rank == 0

# ============================================================================
# Alignment Builder
# ============================================================================
def _build_alignment_groups_from_ids(student_token_ids, teacher_token_ids,student_tokenizer,teacher_tokenizer):
        """
        Build alignment groups using a greedy substring-equality algorithm on decoded token pieces.

        Args:
            student_token_ids: List[int]
            teacher_token_ids: List[int]

        Returns:
            Tuple[List[List[int]], List[List[int]]]: student and teacher alignment groups
        """

        def to_canonical_pieces(tok, ids):
            pieces = []
            prev = ""
            for k in range(len(ids)):
                # IMPORTANT: Do NOT skip special tokens - we need to align them too
                cur = tok.decode(
                    ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                # Extract the incremental addition (may include spaces/ZWJ/etc.)
                pieces.append(cur[len(prev) :])
                prev = cur
            return pieces

        s_pieces = to_canonical_pieces(student_tokenizer, student_token_ids)
        t_pieces = to_canonical_pieces(teacher_tokenizer, teacher_token_ids)

        i = j = 0
        s_buf = t_buf = ""
        s_group = []
        t_group = []
        s_groups = []
        t_groups = []

        def flush():
            if s_group and t_group:
                s_groups.append(s_group.copy())
                t_groups.append(t_group.copy())

        # Greedily accumulate pieces until substrings match, then flush
        while i < len(s_pieces) or j < len(t_pieces):
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf = t_buf = ""
                s_group = []
                t_group = []
                continue

            if s_buf == "" and i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
                continue
            if t_buf == "" and j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
                continue

            if len(s_buf) <= len(t_buf):
                if i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
            else:
                if j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1

        # Flush any remainder if both sides accumulated something
        if s_buf == t_buf and s_group and t_group:
            flush()
        elif s_group or t_group:
            # Handle remaining unmatched tokens by forcing a flush
            # This ensures both sides have the same number of alignment groups
            if s_group or t_group:
                # Ensure both groups have content (even if empty list)
                if not s_group:
                    s_group = []
                if not t_group:
                    t_group = []
                # Force flush even if buffers don't match
                if s_group or t_group:
                    s_groups.append(s_group.copy() if s_group else [])
                    t_groups.append(t_group.copy() if t_group else [])

        return s_groups, t_groups

def build_alignment_groups(text, student_tok, teacher_tok):
    """Build token groups that decode to the same substring."""
    s_ids = student_tok.encode(text, add_special_tokens=False)
    t_ids = teacher_tok.encode(text, add_special_tokens=False)
    s_groups, t_groups = _build_alignment_groups_from_ids(s_ids, t_ids, student_tok, teacher_tok)

    return s_ids, t_ids, s_groups, t_groups


def expand_student_tokens(student_ids, student_tok, teacher_tok):
    """Per student token → list of teacher token ids via decode + re-encode."""
    expand_map = []
    for sid in student_ids:
        s = student_tok.decode([sid], skip_special_tokens=True)
        t_ids = teacher_tok.encode(s, add_special_tokens=False)
        if not t_ids:
            t_ids = [teacher_tok.unk_token_id or 0]
        expand_map.append(t_ids)
    return expand_map

# ============================================================================
# [OPT-3] Super-sequence builder + single-pass teacher forward
# ============================================================================
def build_single_supersequence(
    student_ids: List[int],
    teacher_ids: List[int],
    s_groups: List[List[int]],
    t_groups: List[List[int]],
    expand_map: List[List[int]],
    max_seq_len: int = 2048,
) -> Tuple[List[int], torch.Tensor, List[int], List[int]]:
    """
    Compress N teacher sequences into 1 super-sequence + 2D attention mask.
 
    Super-sequence layout (per group g):
        [group_g original teacher prefix tokens] [exp(s_i) for each s_i in group g]
 
    Attention mask ensures each extraction point sees exactly the same context
    as the original per-token sequence would have provided.
 
    Returns:
        super_tokens:    List[int]   - token ids for the super-sequence
        attn_mask:       Tensor (L, L) bool - True = can attend
        extract_pos:     List[int]   - positions to extract hidden states from
        valid_si:        List[int]   - which student indices these correspond to
    """
    # Build student_idx -> (group_idx, position_in_group)
    s2g = {}
    s2p = {}
    for gi, sg in enumerate(s_groups):
        for p, si in enumerate(sg):
            s2g[si] = gi
            s2p[si] = p
 
    # === Lay out the super-sequence ===
    super_tokens = []
    tok_group = []        # group index for each position
    tok_is_expand = []    # True if expand token, False if prefix token
    tok_owner_si = []     # student idx if expand, -1 if prefix
    tok_ingroup_pos = []  # position-in-group of the owning student token, -1 if prefix
 
    extract_positions = {}  # student_idx -> position in super_seq
 
    for gi in range(len(s_groups)):
        # Group prefix: original teacher tokens
        for ti in t_groups[gi]:
            super_tokens.append(teacher_ids[ti])
            tok_group.append(gi)
            tok_is_expand.append(False)
            tok_owner_si.append(-1)
            tok_ingroup_pos.append(-1)
 
        # Each student token's expand in this group
        for si in s_groups[gi]:
            p_in_g = s2p[si]
            for tok_id in expand_map[si]:
                super_tokens.append(tok_id)
                tok_group.append(gi)
                tok_is_expand.append(True)
                tok_owner_si.append(si)
                tok_ingroup_pos.append(p_in_g)
            # Extract from the last token of this expand
            extract_positions[si] = len(super_tokens) - 1
 
    L = len(super_tokens)
 
    # Truncation fallback: if too long, we can't do single-pass.
    # Return empty to signal caller should fall back to original method.
    if L > max_seq_len * 2:
        return [], None, [], []
 
    # === Build 2D attention mask (vectorized) ===
    t_group_t = torch.tensor(tok_group, dtype=torch.long)
    t_is_exp = torch.tensor(tok_is_expand, dtype=torch.bool)
    t_is_pfx = ~t_is_exp
    t_igp = torch.tensor(tok_ingroup_pos, dtype=torch.long)  # in-group position
 
    # (L,1) vs (1,L) broadcasting for pairwise comparisons
    gi_q = t_group_t.unsqueeze(1)   # query groups  (L, 1)
    gi_k = t_group_t.unsqueeze(0)   # key groups    (1, L)
    exp_k = t_is_exp.unsqueeze(0)   # key is expand (1, L)
    pfx_k = t_is_pfx.unsqueeze(0)   # key is prefix (1, L)
    igp_q = t_igp.unsqueeze(1)      # query in-group pos (L, 1)
    igp_k = t_igp.unsqueeze(0)      # key in-group pos   (1, L)
 
    pos = torch.arange(L)
    causal = pos.unsqueeze(0).T >= pos.unsqueeze(0)  # (L, L), causal[i,j] = (i >= j)
 
    # --- Rules for PREFIX query positions ---
    # Can see: earlier groups (all) | same group prefix (causal only)
    pfx_rule = ((gi_k < gi_q) & pfx_k) | ((gi_k == gi_q) & pfx_k & causal)
 
    # --- Rules for EXPAND query positions ---
    # Can see:
    #   1) All tokens from earlier groups
    #   2) Same group, expand, earlier in-group position (all tokens of that expand)
    #   3) Same group, expand, same in-group position, causal
    #   4) Same group PREFIX: NOT visible (matches original code's prefix logic)
    exp_rule = (
        ((gi_k < gi_q) & pfx_k)
        | ((gi_k == gi_q) & exp_k & (igp_k < igp_q))
        | ((gi_k == gi_q) & exp_k & (igp_k == igp_q) & causal)
    )
 
    # Combine: use prefix rule for prefix queries, expand rule for expand queries
    is_pfx_q = t_is_pfx.unsqueeze(1).expand(L, L)
    attn_mask = torch.where(is_pfx_q, pfx_rule, exp_rule)
 
    # === Collect extraction positions in student_ids order ===
    extract_pos_list = []
    valid_si_list = []
    for si in range(len(student_ids)):
        if si in extract_positions:
            extract_pos_list.append(extract_positions[si])
            valid_si_list.append(si)
 
    return super_tokens, attn_mask, extract_pos_list, valid_si_list
 
 
@torch.no_grad()
def teacher_forward_superseq(
    model: nn.Module,
    super_seq: List[int],
    attn_mask_2d: torch.Tensor,
    extract_positions: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    [OPT-3] Single forward pass on the super-sequence.
    
    Returns: (N, hidden_dim) float32 tensor, one hidden state per extraction point.
    """
    model.eval()
    L = len(super_seq)
 
    input_ids = torch.tensor([super_seq], dtype=torch.long, device=device)  # (1, L)
 
    # Convert bool mask to 4D float attention mask:
    # HuggingFace convention: 0.0 = attend, -inf = masked
    # Shape: (1, 1, L, L) — broadcast over batch and heads
    mask_4d = attn_mask_2d.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L,L) bool
    attn_mask_float = torch.where(
        mask_4d,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float('-inf'), dtype=dtype, device=device),
    )
 
    with torch.amp.autocast("cuda", dtype=dtype):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask_float,
            output_hidden_states=True,
            use_cache=False,
        )
 
    last_hidden = outputs.hidden_states[-1]  # (1, L, hdim)
    positions = torch.tensor(extract_positions, dtype=torch.long, device=device)
    extracted = last_hidden[0, positions].float()  # (N, hdim)
 
    del outputs, last_hidden, input_ids, mask_4d, attn_mask_float
    torch.cuda.empty_cache()
 
    return extracted

# ============================================================================
# Original teacher forward (kept as fallback for long sequences)
# ============================================================================
def build_teacher_sequences_optimized(
    student_ids, teacher_ids, s_groups, t_groups, expand_map, max_seq_len=2048,
):
    """Original per-token sequence builder — used as fallback when super-seq is too long."""
    s2g = {}
    for gi, sg in enumerate(s_groups):
        for si in sg:
            s2g[si] = gi
 
    group_prefix_cache = [[]]
    cumulative = []
    for gi in range(len(t_groups)):
        for ti in t_groups[gi]:
            cumulative.append(teacher_ids[ti])
        group_prefix_cache.append(list(cumulative))
 
    all_seqs = []
    all_tgt_pos = []
 
    for si in range(len(student_ids)):
        gi = s2g.get(si, -1)
        if gi < 0:
            continue
        prefix = group_prefix_cache[gi]
        sg = s_groups[gi]
        pos_in_group = sg.index(si)
        within_prefix = []
        if pos_in_group > 0:
            for ii in range(0, pos_in_group):
                within_prefix += expand_map[si + ii - pos_in_group]
        expanded = expand_map[si]
        seq = prefix + within_prefix + expanded
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        all_seqs.append(seq)
        all_tgt_pos.append(len(seq) - 1)
 
    return all_seqs, all_tgt_pos
 
 
@torch.no_grad()
def teacher_forward_batched(
    model, sequences, target_positions, batch_size, device, dtype,
):
    """Original batched forward — used as fallback."""
    model.eval()
    N = len(sequences)
    if N == 0:
        return torch.zeros(0)
    hidden_dim = model.config.hidden_size
    indices = sorted(range(N), key=lambda i: len(sequences[i]))
    sorted_seqs = [sequences[i] for i in indices]
    sorted_tgts = [target_positions[i] for i in indices]
    out = torch.zeros(N, hidden_dim, dtype=torch.float32, device=device)
 
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_seqs = sorted_seqs[start:end]
        batch_tgts = sorted_tgts[start:end]
        bsz = len(batch_seqs)
        max_len = max(len(s) for s in batch_seqs)
        input_ids = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        attn_mask = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        adjusted_tgts = []
        for i, seq in enumerate(batch_seqs):
            pad = max_len - len(seq)
            input_ids[i, pad:] = torch.tensor(seq, dtype=torch.long, device=device)
            attn_mask[i, pad:] = 1
            adjusted_tgts.append(batch_tgts[i] + pad)
        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = model(
                input_ids=input_ids, attention_mask=attn_mask,
                output_hidden_states=True, use_cache=False,
            )
        last_hidden = outputs.hidden_states[-1]
        for i in range(bsz):
            out[start + i] = last_hidden[i, adjusted_tgts[i]].float()
        del outputs, last_hidden, input_ids, attn_mask
        torch.cuda.empty_cache()
 
    result = torch.zeros_like(out)
    for new_idx, orig_idx in enumerate(indices):
        result[orig_idx] = out[new_idx]
    return result

# ============================================================================
# [OPT-3] Unified teacher hidden state extraction
# ============================================================================
def get_teacher_hidden_states(
    sample: dict,
    model: nn.Module,
    max_seq_len: int,
    teacher_bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Get teacher hidden states for all student tokens in a sample.
    
    Tries the optimized single super-sequence path first [OPT-3].
    Falls back to the original N-sequence batched path if the super-sequence
    exceeds max_seq_len.
    
    Returns: (n, hidden_dim) tensor where n = len(student_ids)
    """
    s_ids = sample["s_ids"]
    t_ids = sample["t_ids"]
    sg = sample["sg"]
    tg = sample["tg"]
    em = sample["em"]
    n = len(s_ids)
    hdim = model.config.hidden_size
 
    # --- Try super-sequence path ---
    super_tokens, attn_mask, extract_pos, valid_si = build_single_supersequence(
        s_ids, t_ids, sg, tg, em, max_seq_len
    )
 
    if super_tokens and len(super_tokens) > 0:
        # Super-sequence fits within max_seq_len: single forward pass
        hidden = teacher_forward_superseq(
            model, super_tokens, attn_mask, extract_pos, device, dtype
        )
        # Map back to full student_ids ordering
        result = torch.zeros(n, hdim, dtype=torch.float32, device=device)
        for idx, si in enumerate(valid_si):
            result[si] = hidden[idx]
        return result
 
    # --- Fallback: original N-sequence batched path ---
    seqs, tgt_pos = build_teacher_sequences_optimized(
        s_ids, t_ids, sg, tg, em, max_seq_len
    )
    if not seqs:
        return torch.zeros(n, hdim, dtype=torch.float32, device=device)
 
    return teacher_forward_batched(model, seqs, tgt_pos, teacher_bsz, device, dtype)


# ============================================================================
# Projection Head
# ============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.weight)


# ============================================================================
# Dataset
# ============================================================================

class PreprocessedDataset(Dataset):
    def __init__(self, texts, student_tok, teacher_tok, max_tokens, if_write_cache, cache_path=None):
        self._data = []
        if not cache_path:
            raise ValueError("cache_path must not be set !!!")
        
        if not if_write_cache:
            cache_path = Path(cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(f"cache file does not exist: {cache_path}, 'if_write_cache' should be set 'True'.")

            time_start = time.time()
            with open(cache_path, "rb") as f:
                self._data = pickle.load(f)
            logger.info(f"Loaded {len(self._data)} samples from cache in {time.time() - time_start:.1f}s")
        else:
            for text in tqdm(texts, desc="Preprocessing"):
                try:
                    s_ids, t_ids, sg, tg = build_alignment_groups(text, student_tok, teacher_tok)
                    if not s_ids or not t_ids:
                        continue
                    if len(s_ids) > max_tokens or len(t_ids) > max_tokens:
                        continue

                    em = expand_student_tokens(s_ids, student_tok, teacher_tok)
                    self._data.append({
                        "s_ids": s_ids, "t_ids": t_ids,
                        "sg": sg, "tg": tg, "em": em,
                    })
                except Exception:
                    continue
            with open(cache_path, "wb") as f:
                pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cache saved to {cache_path}")
        logger.info(f"Dataset: {len(self._data)}/{len(texts)} samples ready")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]
    
    def __getitems__(self, indices):
        return [self._data[i] for i in indices]


# ============================================================================
# Training
# ============================================================================

@dataclass
class Config:
    student_model: str = "meta-llama/Llama-3.2-1B"       # only tokenizer is loaded
    teacher_model: str = "Qwen/Qwen2.5-1.5B"
    dataset_name: str = "HuggingFaceTB/smollm-corpus"
    dataset_subset: str = "cosmopedia-v2"
    text_col: str = "text"
    max_samples: int = 10000
    max_tokens: int = 512
    max_seq_len: int = 2048
    batch_size: int = 4
    teacher_bsz: int = 32
    lr: float = 1e-3
    wd: float = 0.01
    epochs: int = 3
    warmup: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int = 4
    dtype: str = "bfloat16"
    log_every: int = 10
    save_every: int = 500
    output_dir: str = "output_projection"
    if_write_cache: bool = False
    cache_path: str = None


def run(cfg: Config):
    rank, world_size, local_rank = setup_ddp()
    dev = torch.device(f"cuda:{local_rank}")
    
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.dtype]
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Tokenizers (student model is NOT loaded, only its tokenizer)
    s_tok = AutoTokenizer.from_pretrained(cfg.student_model)
    t_tok = AutoTokenizer.from_pretrained(cfg.teacher_model)
    for tok in (s_tok, t_tok):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    # Student vocab size from tokenizer (no need to load the model)
    s_vocab = s_tok.vocab_size
    if is_main(rank):
        logger.info(f"Student vocab size (from tokenizer): {s_vocab}")

    # Teacher model (frozen, only need hidden states)
    if is_main(rank):
        logger.info("Loading teacher model (frozen)...")
    t_model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model, torch_dtype=dt, 
    ).to(dev) 
    t_model.eval()
    for p in t_model.parameters():
        p.requires_grad = False

    t_hdim = t_model.config.hidden_size
    if is_main(rank):
        logger.info(f"Projection: {t_hdim} → {s_vocab}")

    proj = ProjectionHead(t_hdim, s_vocab).to(dev)
    proj_ddp = DDP(proj, device_ids=[local_rank], find_unused_parameters=False)

    # Data
    if is_main(rank):
        logger.info("Loading data...")
    ds = Dataset.from_parquet(cfg.dataset_name)
    texts = []
    for i, x in enumerate(ds):
        if i >= cfg.max_samples and cfg.max_samples != -1:
            break
        t = x.get(cfg.text_col, "")
        if len(t.strip()) > 20:
            texts.append(t)
            
    cache = cfg.cache_path if cfg.cache_path else None
    
    dataset = PreprocessedDataset(texts, s_tok, t_tok, cfg.max_tokens, cfg.if_write_cache,
                              cache_path=cache)

    
    cleanup_ddp()


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--student_model", default="meta-llama/Llama-3.2-1B",
                     help="Student model name (only tokenizer is loaded)")
    pa.add_argument("--teacher_model", default="Qwen/Qwen2.5-1.5B")
    pa.add_argument("--dataset_name", default="HuggingFaceTB/smollm-corpus")
    pa.add_argument("--dataset_subset", default="cosmopedia-v2")
    pa.add_argument("--text_col", default="text")
    pa.add_argument("--max_samples", type=int, default=-1,help="The number of samples from source data (before preprocessed). -1 means 'all'. ")
    pa.add_argument("--max_tokens", type=int, default=512, help="The max tokens of each sample. It only works when writing the pickle cache file.")
    pa.add_argument("--max_seq_len", type=int, default=2048)
    pa.add_argument("--batch_size",  type=int,   default=4,help="Per-GPU batch size")
    pa.add_argument("--teacher_bsz", type=int, default=32)
    pa.add_argument("--lr", type=float, default=1e-5)
    pa.add_argument("--epochs", type=int, default=1)
    pa.add_argument("--grad_accum", type=int, default=1)
    pa.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    pa.add_argument("--output_dir", default="output_projection")
    pa.add_argument("--if_write_cache",  action="store_true", help="If set, the preprocessing stage will start and a pickle file will be written.")
    pa.add_argument("--cache_path", default=None,
                    help="The processed data path. Must be set.")
    args = pa.parse_args()

    run(Config(
        student_model=args.student_model, teacher_model=args.teacher_model,
        dataset_name=args.dataset_name, dataset_subset=args.dataset_subset,
        text_col=args.text_col, max_samples=args.max_samples, max_tokens=args.max_tokens,
        batch_size=args.batch_size, teacher_bsz=args.teacher_bsz, lr=args.lr,
        epochs=args.epochs, grad_accum=args.grad_accum, dtype=args.dtype,
        output_dir=args.output_dir,if_write_cache=args.if_write_cache,cache_path = args.cache_path
    ))
    
# torchrun --nproc_per_node=1 data_preprocess.py --student_model /inspire/hdd/project/smarteducation/public/models/Llama-3.2-1B-Instruct --teacher_model /inspire/hdd/project/smarteducation/public/models/Qwen3-4B-Instruct --dataset_name /inspire/dataset/nemotron-cc-v2/v1/Diverse-QA/part_000001.parquet --output_dir /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/output --batch_size 48 --max_tokens 4096 --max_seq_len 4096 --max_samples -1 --text_col text --cache_path /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/pretrain_nemotron_diverseQA_part2.pkl --if_write_cache