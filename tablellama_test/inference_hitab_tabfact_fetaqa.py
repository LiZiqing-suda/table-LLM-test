# 完全删除llama_attn_replace相关导入（核心修复）
import os
import json
import re
import sys
import math
import torch
import argparse
import transformers
from peft import PeftModel
from transformers import GenerationConfig
from supervised_fine_tune import PROMPT_DICT
from tqdm import tqdm
import random

random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed_all(333)



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='已禁用，无实际作用')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    parser.add_argument('--input_data_file', type=str, default='input_data/', help='')
    parser.add_argument('--output_data_file', type=str, default='output_data/', help='')
    args = parser.parse_args()
    return args


def generate_prompt(instruction, question, input_seg=None):
    if input_seg is not None and input_seg != "":
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def build_generator(
    item, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=False  # 核心：默认use_cache=False
):
    def response(item):
        prompt = generate_prompt(instruction=item["instruction"], input_seg=item["input_seg"], question=item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 核心修复：禁用use_cache，避免传递past_key_values参数
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            past_key_values=None,  # 显式置空，彻底避免参数传递
            do_sample=True,  # 显式开启采样（避免默认参数冲突）
        )
        out = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # 兼容不同prompt分割方式
        if prompt in out:
            out = out.split(prompt)[1].strip()
        return out

    return response


def main(args):
    # 完全删除flash_attn替换逻辑（核心修复）
    # if args.flash_attn:
    #     replace_llama_attn()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 纯原生模型加载（无量化、无flash_attn替换）
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(32001)

    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length = args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="left",
        use_fast=False,
    )
    # 补充：设置pad_token（Llama默认无pad_token，避免生成报错）
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 模型评估模式 + 禁用梯度
    model.eval()
    torch.no_grad()

    # 加载测试数据
    with open(args.input_data_file, "r") as f:
        test_data = json.load(f)
    
    # 推理预测（先处理前5条验证）
    test_data_pred = []
    #process_num = min(50, len(test_data))
    right = []
    process_num = len(test_data)
    tot = 0
    r = 0
    for i in tqdm(range(process_num)):
        tot+=1
        item = test_data[i]
        new_item = {}
        respond = build_generator(
            item, model, tokenizer, 
            temperature=args.temperature, 
            top_p=args.top_p,
            max_gen_len=args.max_gen_len,
            use_cache=False  # 强制禁用use_cache
        )
        output = respond(item)

        new_item["idx"] = i
        new_item["instruction"] = test_data[i]["instruction"]
        new_item["input_seg"] = test_data[i]["input_seg"]
        new_item["question"] = test_data[i]["question"]
        new_item["output"] = test_data[i]["output"]
        new_item["predict"] = output
        if output==test_data[i]["output"]:
            r+=1
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_data_file), exist_ok=True)
    with open(args.output_data_file, "w", encoding="utf-8") as f:
        json.dump(test_data_pred, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 推理完成！共处理{process_num}条数据，结果已保存到：{args.output_data_file}")
    print(r)
    print(tot)
    print(r/tot)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)
    args = parse_config()
    main(args)