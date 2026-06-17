import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
import argparse

def generate_teacher_new_head(
    teacher_model,
    qwen_tokenizer,
    llama_tokenizer,
    messages,
    max_new_tokens=256,
    device="cuda",
):
    teacher_model.eval()

    # 1. Qwen chat template 生成 prompt 文本
    prompt_text = llama_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    generated_text = ""

    for step in range(max_new_tokens):
        # 2. 每一步都用 Qwen tokenizer 重新 encode prompt + 当前生成文本
        full_text = prompt_text + generated_text

        inputs = qwen_tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
            logits = outputs.logits[:, -1, :]  # Llama vocab logits

        # 3. 在 Llama vocab 上选下一个 token
        next_llama_id = torch.argmax(logits, dim=-1).item()

        # 4. 用 Llama tokenizer decode 这个 token
        piece = llama_tokenizer.decode(
            [next_llama_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # 5. 拼接自然语言文本
        generated_text += piece

        # 6. 停止条件
        if next_llama_id in [
            llama_tokenizer.eos_token_id,
            llama_tokenizer.pad_token_id,
        ]:
            break

        if piece in ["<|end_of_text|>", "<|eot_id|>"]:
            break

    return prompt_text, generated_text

def main(teacher_model_name_or_path, new_head_path, device="cuda"):
    # -----------------------------
    # 1. 加载 Qwen teacher backbone
    # -----------------------------
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name_or_path,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device)
    
    # -----------------------------
    # 2. 替换 Llama head
    # -----------------------------
    head_weight = torch.load(new_head_path, map_location="cpu")["weight"]
    vocab_size, hidden_size = head_weight.shape
    # 确认 hidden_size 与 teacher_model hidden size 匹配
    assert hidden_size == teacher_model.config.hidden_size, f"{hidden_size} != {teacher_model.config.hidden_size}"
    
    model_dtype = next(teacher_model.parameters()).dtype
    model_device = next(teacher_model.parameters()).device
    
    new_lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False,dtype=model_dtype,device=model_device,).to(device)
    new_lm_head.weight.data.copy_(head_weight.to(teacher_model.dtype))
    teacher_model.lm_head = new_lm_head
    teacher_model.eval()

    # -----------------------------
    # 3. Tokenizers
    # -----------------------------
    # Qwen tokenizer 用于生成 teacher input_ids
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name_or_path, local_files_only=True)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Llama tokenizer 用于 decode teacher logits
    llama_tokenizer = PreTrainedTokenizerFast.from_pretrained("/inspire/hdd/project/smarteducation/public/models/Llama-3.2-1B-Instruct", local_files_only=True)
    
    # -----------------------------
    # 4. 构建 Countdown prompt
    # -----------------------------
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
        },
        {
            "role": "user",
            "content": "Using the numbers [56, 2, 69], create an equation that equals 82. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags."
        }
    ]

    # encode prompt with Qwen tokenizer
    prompt_text = "".join([f"<|{m['role']}|>{m['content']}" for m in messages])
    input_ids = teacher_tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    # -----------------------------
    # 5. Teacher forward
    # -----------------------------
    with torch.no_grad():
        outputs = teacher_model(input_ids=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # -----------------------------
    # 6. Decode top-k token at last position
    # -----------------------------
    last_logits = logits[0, -1]  # shape [vocab_size]
    topk = 30
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, topk)

    print("Top tokens at last position:")
    for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
        token_str = llama_tokenizer.decode([idx], skip_special_tokens=False)
        print(f"{idx}: {token_str} ({prob:.4f})")
    
    # -----------------------------
    # 7. Greedy generation
    # -----------------------------
    prompt_text, generated_text = generate_teacher_new_head(
    teacher_model=teacher_model,
    qwen_tokenizer=teacher_tokenizer,
    llama_tokenizer=llama_tokenizer,
    messages=messages,
    max_new_tokens=1024,
    device=device,
)
    print(prompt_text)
    print("----------------------------")
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name_or_path", type=str, default="Qwen3-4B-Instruct")
    parser.add_argument("--new_head_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args.teacher_model_name_or_path, args.new_head_path, args.device)