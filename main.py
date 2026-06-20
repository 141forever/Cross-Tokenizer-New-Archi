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

def build_single_supersequence(
    student_ids: List[int],
    teacher_ids: List[int],
    s_groups: List[List[int]],
    t_groups: List[List[int]],
    expand_map: List[List[int]],
    max_seq_len: int = 2048,
) -> Tuple[List[int], Optional[torch.Tensor], List[int], List[int], List[int]]:
    """
    Build a two-block super-sequence:

        P block:
            [all original Qwen teacher tokens]
            = teacher_ids

        E block:
            [all Llama-token pieces expanded into Qwen tokens]
            = expand_map[0] + expand_map[1] + ... + expand_map[n-1]

    The physical token ids are more natural than the old interleaved layout:

        old:
            P_g0, E_g0, P_g1, E_g1, ...

        new:
            P_g0, P_g1, ..., P_gn, E_g0, E_g1, ..., E_gn

    But the attention mask makes each E query see exactly the same logical
    context as the original per-token hybrid-prefix sequence.

    Returns:
        super_tokens:
            List[int], shape L

        attn_mask:
            Bool Tensor, shape (L, L), True means can attend

        position_ids:
            List[int], shape L
            Logical RoPE positions.

        extract_pos:
            List[int], physical positions in super_tokens from which to extract hidden states

        valid_si:
            List[int], student token indices corresponding to extract_pos
    """

    # ------------------------------------------------------------
    # 0. Basic maps
    # ------------------------------------------------------------
    s2g = {}
    s2p = {}
    for gi, sg in enumerate(s_groups):
        for p, si in enumerate(sg):
            s2g[si] = gi
            s2p[si] = p

    num_groups = len(s_groups)
    num_student_tokens = len(student_ids)

    # teacher_prefix_lens[gi] = number of original teacher tokens before group gi
    # This is exactly group_prefix_cache[gi]'s length in your fallback logic.
    teacher_prefix_lens = []
    cur_teacher_len = 0
    for gi in range(num_groups):
        teacher_prefix_lens.append(cur_teacher_len)
        cur_teacher_len += len(t_groups[gi])

    # ------------------------------------------------------------
    # 1. Build P block: all original Qwen teacher tokens
    # ------------------------------------------------------------
    super_tokens = []
    position_ids = []

    tok_group = []          # group index for each physical token
    tok_is_expand = []      # False for P block, True for E block
    tok_owner_si = []       # student index if E token, -1 if P token
    tok_ingroup_pos = []    # position inside s_group if E token, -1 if P token
    tok_expand_offset = []  # offset inside current expanded student token, -1 if P token

    # P positions are original Qwen positions: 0, 1, 2, ...
    # P block is physically and logically the full original Qwen sequence.
    for gi in range(num_groups):
        for local_t_pos, ti in enumerate(t_groups[gi]):
            super_tokens.append(teacher_ids[ti])

            # Since t_groups are built from teacher_ids order,
            # teacher_prefix_lens[gi] + local_t_pos is the original teacher position.
            position_ids.append(teacher_prefix_lens[gi] + local_t_pos)

            tok_group.append(gi)
            tok_is_expand.append(False)
            tok_owner_si.append(-1)
            tok_ingroup_pos.append(-1)
            tok_expand_offset.append(-1)

    p_block_len = len(super_tokens)

    # ------------------------------------------------------------
    # 2. Build E block: all expanded Llama-token pieces, in original student order
    # ------------------------------------------------------------
    extract_positions = {}

    # For each group, we need to know how many expanded Qwen tokens appeared
    # before each student token inside this group.
    #
    # This is used for logical position_ids:
    #   pos(E(si, off)) =
    #       teacher_prefix_len_before_group
    #       + expanded_len_before_si_inside_group
    #       + off
    #
    # This matches the original fallback virtual sequence:
    #   original teacher prefix before group
    #   + expanded previous student tokens inside group
    #   + expanded current student token
    for gi in range(num_groups):
        expanded_len_so_far_in_group = 0

        for si in s_groups[gi]:
            if si < 0 or si >= num_student_tokens:
                continue

            p_in_g = s2p[si]
            expanded = expand_map[si]

            if not expanded:
                continue

            for off, tok_id in enumerate(expanded):
                super_tokens.append(tok_id)

                logical_pos = (
                    teacher_prefix_lens[gi]
                    + expanded_len_so_far_in_group
                    + off
                )
                position_ids.append(logical_pos)

                tok_group.append(gi)
                tok_is_expand.append(True)
                tok_owner_si.append(si)
                tok_ingroup_pos.append(p_in_g)
                tok_expand_offset.append(off)

            # Extract from the last token of this student's expanded piece.
            extract_positions[si] = len(super_tokens) - 1

            expanded_len_so_far_in_group += len(expanded)

    L = len(super_tokens)

    if L == 0:
        return [], None, [], [], []

    # Physical length guard.
    #
    # Your original code used max_seq_len * 2 because supersequence contains
    # both original teacher tokens and expanded tokens.
    if L > max_seq_len * 2:
        return [], None, [], [], []

    # Logical position guard.
    #
    # Even if physical length is okay, if logical RoPE position exceeds max_seq_len,
    # it is safer to fallback to the original path.
    if max(position_ids) + 1 > max_seq_len:
        return [], None, [], [], []

    # ------------------------------------------------------------
    # 3. Vectorized attention mask
    # ------------------------------------------------------------
    t_group = torch.tensor(tok_group, dtype=torch.long)
    t_is_exp = torch.tensor(tok_is_expand, dtype=torch.bool)
    t_is_pfx = ~t_is_exp
    t_igp = torch.tensor(tok_ingroup_pos, dtype=torch.long)
    t_owner_si = torch.tensor(tok_owner_si, dtype=torch.long)

    gi_q = t_group.unsqueeze(1)     # (L, 1)
    gi_k = t_group.unsqueeze(0)     # (1, L)

    is_exp_q = t_is_exp.unsqueeze(1)
    is_pfx_q = t_is_pfx.unsqueeze(1)

    is_exp_k = t_is_exp.unsqueeze(0)
    is_pfx_k = t_is_pfx.unsqueeze(0)

    igp_q = t_igp.unsqueeze(1)
    igp_k = t_igp.unsqueeze(0)

    owner_q = t_owner_si.unsqueeze(1)
    owner_k = t_owner_si.unsqueeze(0)

    pos = torch.arange(L)
    causal = pos.unsqueeze(1) >= pos.unsqueeze(0)

    # ------------------------------------------------------------
    # Rule A: P query tokens
    #
    # P block is the full original Qwen sequence.
    # Prefix tokens should run as a normal causal language-model sequence
    # inside P block only.
    #
    # They should not attend to E block.
    # ------------------------------------------------------------
    pfx_rule = is_pfx_k & causal

    # ------------------------------------------------------------
    # Rule B: E query tokens
    #
    # For an expanded token belonging to student token si in group gi,
    # it should see exactly the original fallback virtual context:
    #
    #   1. P tokens from earlier groups only:
    #        group_k < group_q and key is P
    #
    #      This corresponds to:
    #        original teacher prefix before current group
    #
    #   2. E tokens from the same group and earlier student pieces:
    #        group_k == group_q, key is E, ingroup_pos_k < ingroup_pos_q
    #
    #      This corresponds to:
    #        within_prefix = expanded previous student tokens inside group
    #
    #   3. E tokens from the same student piece causally:
    #        same owner_si and physical causal
    #
    #      This corresponds to:
    #        current expanded token's causal prefix
    #
    # Important:
    #   E query should NOT see:
    #       - P tokens from the same group
    #       - E tokens from previous groups
    #       - E tokens from future groups
    # ------------------------------------------------------------
    e_rule = (
        ((gi_k < gi_q) & is_pfx_k)
        | ((gi_k == gi_q) & is_exp_k & (igp_k < igp_q))
        | ((gi_k == gi_q) & is_exp_k & (owner_k == owner_q) & causal)
    )

    attn_mask = torch.where(is_pfx_q, pfx_rule, e_rule)

    # ------------------------------------------------------------
    # 4. Collect extraction positions
    # ------------------------------------------------------------
    extract_pos_list = []
    valid_si_list = []

    for si in range(num_student_tokens):
        if si in extract_positions:
            extract_pos_list.append(extract_positions[si])
            valid_si_list.append(si)

    return super_tokens, attn_mask, position_ids, extract_pos_list, valid_si_list
 
 
@torch.no_grad()
def teacher_forward_superseq(
    model: nn.Module,
    super_seq: List[int],
    attn_mask_2d: torch.Tensor,
    position_ids: List[int],
    extract_positions: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Single forward pass on the two-block super-sequence.

    Key points:
        1. attention_mask is a custom 4D additive mask.
        2. position_ids are logical positions, not physical positions.
        3. hidden states are extracted from E block positions.
    """

    model.eval()

    L = len(super_seq)

    assert attn_mask_2d.shape == (L, L), (
        f"attn_mask_2d shape {attn_mask_2d.shape} != ({L}, {L})"
    )
    assert len(position_ids) == L, (
        f"len(position_ids)={len(position_ids)} != L={L}"
    )

    input_ids = torch.tensor(
        [super_seq],
        dtype=torch.long,
        device=device,
    )  # (1, L)

    position_ids_tensor = torch.tensor(
        [position_ids],
        dtype=torch.long,
        device=device,
    )  # (1, L)

    # Convert bool mask to 4D additive mask.
    #
    # HuggingFace attention convention:
    #   0.0  = can attend
    #   -inf = masked
    mask_4d = attn_mask_2d.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, L, L)

    attn_mask_float = torch.where(
        mask_4d,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    with torch.amp.autocast("cuda", dtype=dtype):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask_float,
            position_ids=position_ids_tensor,
            output_hidden_states=True,
            use_cache=False,
        )

    last_hidden = outputs.hidden_states[-1]  # (1, L, hidden_dim)

    positions = torch.tensor(
        extract_positions,
        dtype=torch.long,
        device=device,
    )

    extracted = last_hidden[0, positions].float()  # (N, hidden_dim)

    del outputs, last_hidden, input_ids, position_ids_tensor
    del mask_4d, attn_mask_float, positions
    torch.cuda.empty_cache()

    return extracted

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

    First tries the new two-block super-sequence path:

        P block: all original Qwen tokens
        E block: all expanded Llama pieces

    Returns:
        Tensor of shape (n, hidden_dim), where n = len(student_ids).
    """

    s_ids = sample["s_ids"]
    t_ids = sample["t_ids"]
    sg = sample["sg"]
    tg = sample["tg"]
    em = sample["em"]

    n = len(s_ids)
    hdim = model.config.hidden_size

    # --- New two-block super-sequence path ---
    super_tokens, attn_mask, position_ids, extract_pos, valid_si = build_single_supersequence(
        s_ids,
        t_ids,
        sg,
        tg,
        em,
        max_seq_len,
    )

    if super_tokens and len(super_tokens) > 0:
        hidden = teacher_forward_superseq(
            model=model,
            super_seq=super_tokens,
            attn_mask_2d=attn_mask,
            position_ids=position_ids,
            extract_positions=extract_pos,
            device=device,
            dtype=dtype,
        )

        result = torch.zeros(
            n,
            hdim,
            dtype=torch.float32,
            device=device,
        )

        for idx, si in enumerate(valid_si):
            result[si] = hidden[idx]

        return result

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.weight)

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
    s_vocab = len(s_tok)
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
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                             shuffle=True, drop_last=True)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                     collate_fn=lambda b: b, num_workers=0)

    # Optimizer
    opt = torch.optim.AdamW(proj_ddp.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    total_steps = (len(loader) * cfg.epochs) // cfg.grad_accum
    warmup_steps = int(total_steps * cfg.warmup)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    if is_main(rank):
        logger.info(f"Training: {cfg.epochs} epochs, {total_steps} optimizer steps")

    step = 0
    loss_accum = 0.0
    loss_count = 0

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        for bi, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}",disable=not is_main(rank))):
            batch_loss = torch.tensor(0.0, device=dev)
            valid = 0

            for sample in batch:
                s_ids = sample["s_ids"]
                t_ids = sample["t_ids"]
                n = len(s_ids)
                if n < 2:  # need at least 2 tokens for NTP
                    continue

                t_hidden = get_teacher_hidden_states(
                    sample, t_model, cfg.max_seq_len, cfg.teacher_bsz, dev, dt
                )  # (n, hdim)
                t_hidden = t_hidden.to(dev)
 
                assert t_hidden.shape[0] == n

                # Projection → NTP Cross-Entropy
                # proj_logits[i] predicts student_ids[i+1]
                proj_logits = proj_ddp(t_hidden.detach())  # (n, s_vocab), WITH grad

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
                nn.utils.clip_grad_norm_(proj_ddp.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if is_main(rank) and step % cfg.log_every == 0:
                    avg = loss_accum / max(loss_count, 1)
                    logger.info(f"[Step {step}/{total_steps}] loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")
                    loss_accum = 0.0
                    loss_count = 0

                if is_main(rank) and step % cfg.save_every == 0:
                    p = os.path.join(cfg.output_dir, f"proj_step{step}.pt")
                    torch.save(proj_ddp.module.state_dict(), p)
                    logger.info(f"Saved → {p}")
        
        if is_main(rank):
            p = os.path.join(cfg.output_dir, f"proj_epoch{epoch+1}.pt")
            torch.save(proj_ddp.module.state_dict(), p)
            logger.info(f"Epoch {epoch+1} done → {p}")

    if is_main(rank):
        p = os.path.join(cfg.output_dir, "proj_final.pt")
        torch.save(proj_ddp.module.state_dict(), p)
        logger.info(f"Done → {p}")
    
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
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size, teacher_bsz=args.teacher_bsz, lr=args.lr,
        epochs=args.epochs, grad_accum=args.grad_accum, dtype=args.dtype,
        output_dir=args.output_dir,if_write_cache=args.if_write_cache,cache_path = args.cache_path
    ))
    
# torchrun --nproc_per_node=4 main.py --student_model /inspire/hdd/project/smarteducation/public/models/Llama-3.2-1B-Instruct --teacher_model /inspire/hdd/project/smarteducation/public/models/Qwen3-4B-Instruct --dataset_name /inspire/dataset/nemotron-cc-v2/v1/Diverse-QA/part_000000.parquet --output_dir /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/output --batch_size 48 --max_tokens 4096 --max_seq_len 4096 --max_samples -1 --text_col text --cache_path /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/pretrain_nemotron_diverseQA_part1.pkl

# CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 main.py --student_model /inspire/hdd/project/smarteducation/public/models/Llama-3.2-1B-Instruct --teacher_model /inspire/hdd/project/smarteducation/public/models/Qwen3-4B-Instruct --dataset_name /inspire/dataset/nemotron-cc-v2/v1/Diverse-QA/part_000000.parquet --output_dir /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/output --batch_size 48 --max_tokens 4096 --max_seq_len 4096 --max_samples -1 --text_col text --cache_path /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/pretrain_nemotron_diverseQA_part1.pkl

# python main.py --student_model /inspire/hdd/project/smarteducation/public/models/Llama-3.2-1B-Instruct --teacher_model /inspire/hdd/project/smarteducation/public/models/Qwen3-4B-Instruct --dataset_name /inspire/dataset/nemotron-cc-v2/v1/Diverse-QA/part_000000.parquet --output_dir /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/output --batch_size 48 --max_tokens 4096 --max_seq_len 4096 --max_samples -1 --text_col text --cache_path /inspire/hdd/project/smarteducation/chenkedi-253108120128/Cross-Tokenizer-New-Archi/pretrain_nemotron_diverseQA_part1.pkl