import json
import os
import sys

import torch
from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from prompter import DefaultPrompter


def sys_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


def load_model(base_model, local_files_only=False, lora_weights=""):
    if not local_files_only:
        sys_proxy()
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
        cache_dir="./"
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_files_only,
        cache_dir="./"
    )
    if os.path.exists(lora_weights):
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
    model.eval()
    # model.cuda()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model


def generation(prompt, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=32,
        )
    # print(generation_output)
    s = generation_output[0]
    output = tokenizer.decode(s)
    return output


def main(
        schema_path="data/multiwoz/data/MultiWOZ_2.2/schema.json",
        # base_model="NousResearch/Llama-2-7b-chat-hf",
        base_model="daryl149/llama-2-7b-chat-hf",
        lora_weights="output_model_full",
        # lora_weights="output_model_full/checkpoints-100",
        processed_data_path="data/MultiWOZ_2.2_preprocess/test.json",
        output_file="data/MultiWOZ_2.2_preprocess/test_out.json"
):
    prompter = DefaultPrompter(schema_path)
    tokenizer, model = load_model(base_model, lora_weights=lora_weights)
    data = json.load(open(processed_data_path, "r"))
    response_list = []
    aga_num = 0

    jga_num = 0
    jga_tot = -1
    last_index_turn = ""
    last_full_state = True

    for idx, sample in enumerate(tqdm(data)):
        this_index_turn = f'{sample["index"]}|{sample["turn"]}'
        if last_index_turn and last_index_turn != this_index_turn:
            last_index_turn = this_index_turn
            jga_tot += 1
            if last_full_state:
                jga_num += 1
            last_full_state = True

        if idx > 0 and idx % 100 == 0:
            print(f"AGA for 0~{idx}: {aga_num / (idx + 1)}")
            print(f"JGA for 0~{idx}: {jga_num / jga_tot}")

        prompt = prompter.generate_prompt(sample["dialogue"], sample["domain"], sample["slot"])
        output = generation(prompt, tokenizer, model)
        response = prompter.get_response(output)

        if sample['value'] == response:
            aga_num += 1
        else:
            last_full_state = False
            # print(f"{aga_num/(idx+1)}|||{sample['slot']}|||{sample['value']}|||{response}")

        response_list.append({
            "index": sample["index"],
            "turn": sample["turn"],
            "domain": sample["domain"],
            "slot": sample["slot"],
            # "active": sample["active"],
            "value": response,
            "ground_truth": sample["value"]
        })
    print("accuracy:", 1 - aga_num / len(data))
    json.dump(response_list, open(output_file, "w"))


if __name__ == "__main__":
    main()
