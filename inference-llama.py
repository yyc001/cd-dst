import json
import os
import sys

import torch
from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from contra_llama import PromptContraDecodeLlama
from prompter import DefaultPrompter, Prompter
from llama import Llama


def sys_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


def main(
        schema_path="data/multiwoz/data/MultiWOZ_2.2/schema.json",
        processed_data_path="data/MultiWOZ_2.2_preprocess/test.json",
        output_file="data/MultiWOZ_2.2_preprocess/test_out.json",
        contra_decode=False,
):
    if contra_decode:
        generator = PromptContraDecodeLlama.build(
            ckpt_dir="../llama/llama-2-7b-chat/",
            tokenizer_path="../llama/tokenizer.model",
            max_seq_len=1024,
            max_batch_size=4,
        )
    else:
        generator = Llama.build(
            ckpt_dir="../llama/llama-2-7b-chat/",
            tokenizer_path="../llama/tokenizer.model",
            max_seq_len=1024,
            max_batch_size=4,
        )
    prompter = Prompter(schema_path)
    data = json.load(open(processed_data_path, "r"))
    data = [item for item in data if item["value"] != "none"]
    response_list = []
    aga_num = 0

    jga_num = 0
    jga_tot = -1
    last_index_turn = ""
    last_full_state = True

    for idx, sample in enumerate(tqdm(data)):
        this_index_turn = f'{sample["index"]}|{sample["turn"]}'
        if last_index_turn == "":
            last_index_turn = this_index_turn
        if last_index_turn and last_index_turn != this_index_turn:
            last_index_turn = this_index_turn
            jga_tot += 1
            if last_full_state:
                jga_num += 1
            last_full_state = True

        if idx > 0 and idx % 100 == 0:
            print(f"AGA for 0~{idx}: {aga_num / (idx + 1)}")
            print(f"JGA for 0~{idx}: {jga_num / jga_tot}")

        if contra_decode:
            # print(">>-------------")
            prompt, noob_prompt = prompter.generate_prompt(sample["dialogue"], sample["domain"], sample["slot"], pair=True)
            generator.enable_cd = True
            output = generator.chat_completion(
                [[{"role": "user", "content": prompt}],
                 [{"role": "user", "content": noob_prompt}]],  # type: ignore
                max_gen_len=64,
                temperature=0.6,
                top_p=0.9,
            )[0]['generation']['content']
            # print(output)
            # generator.enable_cd = False
            # output2 = generator.chat_completion(
            #     [[{"role": "user", "content": prompt}]],  # type: ignore
            #     max_gen_len=64,
            #     temperature=0.6,
            #     top_p=0.9,
            # )[0]['generation']['content']
            # print(output2)
        else:
            # prompt = prompter.generate_prompt(sample["dialogue"], sample["domain"], sample["slot"])
            prompt, noob_prompt = prompter.generate_prompt(sample["dialogue"], sample["domain"], sample["slot"],
                                                           pair=True)
            generator.enable_cd = False
            results = generator.chat_completion(
                [[{"role": "user", "content": prompt}]],  # type: ignore
                max_gen_len=64,
                temperature=0.6,
                top_p=0.9,
            )
            output = results[0]['generation']['content']

        # print(output)
        response = prompter.get_response(output)

        if sample['value'] == response:
            aga_num += 1
        else:
            last_full_state = False
            # print(sample["dialogue"])
            print(f"{aga_num / (idx + 1)}|||{sample['slot']}|||{sample['value']}|||{response}")

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
