"""
Cross-Tokenizer Projection Head Trainer (Stage 1)
===================================================

Goal: Train a linear projection W: teacher_hidden_dim → student_vocab_size
so that teacher's representations can predict student token sequences (NTP loss).

After training, W replaces the teacher's lm_head, enabling same-vocab KL distillation
in Stage 2 (teacher+W → student).

Pipeline:
1. text → student_tokenizer → student_ids (n tokens, g groups)
   text → teacher_tokenizer → teacher_ids (m tokens, g groups)
   Build alignment groups so s_groups[i] ↔ t_groups[i] decode to same string.

2. Per student token: decode → re-tokenize with teacher_tokenizer → expand_map.

3. Teacher forward with hybrid prefix (original teacher tokens as prefix,
   expanded tokens for current position) → extract last hidden states.

4. proj_logits = W(teacher_hidden)   shape: (n, student_vocab_size)
   labels = student_ids shifted       (standard next-token prediction)
   loss = CrossEntropy(proj_logits[:-1], student_ids[1:])

Only W is trained. Both teacher backbone and student tokenizer are frozen.
Student MODEL is NOT loaded — only its tokenizer is needed.

Optimizations:
- Length-sorted batching to reduce padding waste
- Mixed precision teacher forward
- Gradient accumulation
- Per-sample GPU memory cleanup
"""

import os
import math
import logging
import pdb
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# Optimized: Build teacher sequences with prefix sharing
# ============================================================================

def build_teacher_sequences_optimized(
    student_ids: List[int],
    teacher_ids: List[int],
    s_groups: List[List[int]],
    t_groups: List[List[int]],
    expand_map: List[List[int]],
    max_seq_len: int = 2048,
) -> Tuple[List[List[int]], List[int]]:
    """
    For each student token, build input = teacher_prefix + expanded_tokens.
    
    Prefix rule: for student token ts_i in group g:
        prefix = [original teacher tokens for groups 0..g-1]
        (within-group prefix is NOT expanded, using original teacher tokens)
    Then append expand_map[i] (the expanded teacher tokens for ts_i).
    Target = last position in the sequence.
    """
    # Map student idx → group idx
    s2g = {}
    for gi, sg in enumerate(s_groups):
        for si in sg:
            s2g[si] = gi

    # Precompute cumulative teacher prefix per group
    # group_prefix_end[g] = list of teacher token ids for groups 0..g-1
    group_prefix_cache = [[]]  # group 0 has empty prefix
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

        # Prefix = teacher tokens for completed groups
        prefix = group_prefix_cache[gi]

        # Within-group: add original teacher tokens for student tokens before si
        sg = s_groups[gi]
        tg = t_groups[gi]
        pos_in_group = sg.index(si)
        within_prefix = []
        if pos_in_group > 0:
            for ii in range(0,pos_in_group):
                within_prefix += expand_map[si+ii-pos_in_group]
        expanded = expand_map[si]
        seq = prefix + within_prefix + expanded

        # Truncate from the left if too long
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]

        all_seqs.append(seq)
        all_tgt_pos.append(len(seq) - 1)

    return all_seqs, all_tgt_pos


# ============================================================================
# Batched Teacher Forward with Length-Sorted Packing
# ============================================================================

@torch.no_grad()
def teacher_forward_batched(
    model: nn.Module,
    sequences: List[List[int]],
    target_positions: List[int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Efficient batched teacher forward. Returns (N, hidden_dim) fp32 tensor."""
    model.eval()
    N = len(sequences)
    if N == 0:
        return torch.zeros(0)

    hidden_dim = model.config.hidden_size

    # Sort by length for better packing (less padding waste)
    indices = sorted(range(N), key=lambda i: len(sequences[i]))
    sorted_seqs = [sequences[i] for i in indices]
    sorted_tgts = [target_positions[i] for i in indices]

    # Allocate output buffer
    out = torch.zeros(N, hidden_dim, dtype=torch.float32, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_seqs = sorted_seqs[start:end]
        batch_tgts = sorted_tgts[start:end]
        bsz = len(batch_seqs)

        max_len = max(len(s) for s in batch_seqs)

        # Right-align (left-pad) for causal LM
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
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        last_hidden = outputs.hidden_states[-1]  # (bsz, max_len, hdim)
        for i in range(bsz):
            out[start + i] = last_hidden[i, adjusted_tgts[i]].float()

        del outputs, last_hidden, input_ids, attn_mask
        torch.cuda.empty_cache()

    # Unsort back to original order
    result = torch.zeros_like(out)
    for new_idx, orig_idx in enumerate(indices):
        result[orig_idx] = out[new_idx]

    return result


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
    def __init__(self, texts, student_tok, teacher_tok, max_tokens=4096):
        self._data = []
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


def run(cfg: Config):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    logger.info(f"Student vocab size (from tokenizer): {s_vocab}")

    # Teacher model (frozen, only need hidden states)
    logger.info("Loading teacher model (frozen)...")
    t_model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model, torch_dtype=dt, device_map="auto",
    )
    t_model.eval()
    for p in t_model.parameters():
        p.requires_grad = False

    t_hdim = t_model.config.hidden_size
    logger.info(f"Projection: {t_hdim} → {s_vocab}")

    proj = ProjectionHead(t_hdim, s_vocab).to(dev)

    # Data
    logger.info("Loading data...")
    ds = Dataset.from_parquet(cfg.dataset_name)
    texts = []
    for i, x in enumerate(ds):
        if i >= cfg.max_samples and cfg.max_samples != -1:
            break
        t = x.get(cfg.text_col, "")
        if len(t.strip()) > 20:
            texts.append(t)
            
    dataset = PreprocessedDataset(texts, s_tok, t_tok, cfg.max_tokens)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=lambda b: b, num_workers=0)

    # Optimizer
    opt = torch.optim.AdamW(proj.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    total_steps = (len(loader) * cfg.epochs) // cfg.grad_accum
    warmup_steps = int(total_steps * cfg.warmup)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    logger.info(f"Training: {cfg.epochs} epochs, {total_steps} optimizer steps")

    step = 0
    loss_accum = 0.0
    loss_count = 0

    for epoch in range(cfg.epochs):
        for bi, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            batch_loss = torch.tensor(0.0, device=dev)
            valid = 0

            for sample in batch:
                s_ids = sample["s_ids"]
                t_ids = sample["t_ids"]
                n = len(s_ids)
                if n < 2:  # need at least 2 tokens for NTP
                    continue

                # Teacher hidden states
                seqs, tgt_pos = build_teacher_sequences_optimized(
                    s_ids, t_ids, sample["sg"], sample["tg"],
                    sample["em"], cfg.max_seq_len,
                )
                if not seqs:
                    continue

                t_hidden = teacher_forward_batched(
                    t_model, seqs, tgt_pos, cfg.teacher_bsz, dev, dt
                )  # (n, hdim)

                assert t_hidden.shape[0] == n

                # Projection → NTP Cross-Entropy
                # proj_logits[i] predicts student_ids[i+1]
                proj_logits = proj(t_hidden.detach())  # (n, s_vocab), WITH grad

                logits_for_loss = proj_logits[:-1]  # (n-1, s_vocab)
                labels = torch.tensor(
                    s_ids[1:], dtype=torch.long, device=dev
                )  # (n-1,)

                ntp_loss = F.cross_entropy(logits_for_loss, labels)
                batch_loss = batch_loss + ntp_loss
                valid += 1

                del t_hidden, proj_logits, logits_for_loss, labels
                torch.cuda.empty_cache()

            if valid > 0:
                loss = batch_loss / (valid * cfg.grad_accum)
                loss.backward()
                loss_accum += loss.item() * cfg.grad_accum
                loss_count += 1

            if (bi + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(proj.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    avg = loss_accum / max(loss_count, 1)
                    logger.info(f"[Step {step}/{total_steps}] loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")
                    loss_accum = 0.0
                    loss_count = 0

                if step % cfg.save_every == 0:
                    p = os.path.join(cfg.output_dir, f"proj_step{step}.pt")
                    torch.save(proj.state_dict(), p)
                    logger.info(f"Saved → {p}")

        p = os.path.join(cfg.output_dir, f"proj_epoch{epoch+1}.pt")
        torch.save(proj.state_dict(), p)
        logger.info(f"Epoch {epoch+1} done → {p}")

    p = os.path.join(cfg.output_dir, "proj_final.pt")
    torch.save(proj.state_dict(), p)
    logger.info(f"Done → {p}")


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--student_model", default="meta-llama/Llama-3.2-1B",
                     help="Student model name (only tokenizer is loaded)")
    pa.add_argument("--teacher_model", default="Qwen/Qwen2.5-1.5B")
    pa.add_argument("--dataset_name", default="HuggingFaceTB/smollm-corpus")
    pa.add_argument("--dataset_subset", default="cosmopedia-v2")
    pa.add_argument("--text_col", default="text")
    pa.add_argument("--max_samples", type=int, default=-1)
    pa.add_argument("--max_tokens", type=int, default=512)
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--teacher_bsz", type=int, default=32)
    pa.add_argument("--lr", type=float, default=1e-5)
    pa.add_argument("--epochs", type=int, default=1)
    pa.add_argument("--grad_accum", type=int, default=4)
    pa.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    pa.add_argument("--output_dir", default="output_projection")
    args = pa.parse_args()

    run(Config(
        student_model=args.student_model, teacher_model=args.teacher_model,
        dataset_name=args.dataset_name, dataset_subset=args.dataset_subset,
        text_col=args.text_col, max_samples=args.max_samples, max_tokens=args.max_tokens,
        batch_size=args.batch_size, teacher_bsz=args.teacher_bsz, lr=args.lr,
        epochs=args.epochs, grad_accum=args.grad_accum, dtype=args.dtype,
        output_dir=args.output_dir,
    ))
    
# python main.py --student_model /inspire/hdd/project/smarteducation/public/models/Llama-3.2-1B-Instruct --teacher_model /inspire/hdd/project/smarteducation/public/models/Qwen3-4B-Instruct --dataset_name /inspire/dataset/nemotron-cc-v2/v1/Diverse-QA/part_000000.parquet --output_dir /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/output  --max_samples 100 --text_col text