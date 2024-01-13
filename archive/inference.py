import json
import os
import sys

import torch
from accelerate import Accelerator
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from prompter import DefaultPrompter, Prompter


def sys_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


def load_model(base_model, local_files_only=False, lora_weights=""):
    # if not local_files_only:
    #     sys_proxy()
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        # local_files_only=local_files_only,
        # cache_dir="./"
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        # local_files_only=local_files_only,
        # cache_dir="./"
    )
    if lora_weights and os.path.exists(lora_weights):
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    # else:
    #     model = get_peft_model(
    #         model=model,
    #         peft_config=LoraConfig(
    #             bias="none",
    #             lora_alpha=16,
    #             lora_dropout=0.05,
    #             r=8,
    #             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #             # target_modules=["q_proj", "v_proj"],
    #             task_type="CAUSAL_LM"
    #         )
    #     )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    model.config.pad_token_id = tokenizer.pad_token_id = 0
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
            max_new_tokens=128,
        )
        # generation_output = model.generate(
        #     input_ids=input_ids,
        #     generation_config=GenerationConfig(
        #         temperature=0.02,
        #         top_p=0,
        #         top_k=1,
        #         num_beams=1,
        #     ),
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     max_new_tokens=128,
        # )
    # print(generation_output)
    s = generation_output[0]
    output = tokenizer.decode(s)
    return output


# def main(
#         schema_path="data/multiwoz/data/MultiWOZ_2.2/schema.json",
#         base_model="NousResearch/Llama-2-7b-chat-hf",
#         lora_weights="Checkpoint_files/checkpoint-4000",
#         processed_data_path="data/MultiWOZ_2.2_preprocess/test.json",
#         output_file="data/MultiWOZ_2.2_preprocess/test_out.json"
# ):
def main(
        schema_path="data/multiwoz/data/MultiWOZ_2.2/schema.json",
        base_model="baffo32/decapoda-research-llama-7b-hf",
        lora_weights="LDST/Checkpoint_files/Few-shot_MultiWOZ2-4_10percent",
        # processed_data_path="LDST/Data/MULTIWOZ2.4_preprocess/test_5p.json",
        processed_data_path="data/MultiWOZ_2.4_preprocess/test.json",
        output_file="data/MultiWOZ_2.4_preprocess/test_out_LDST_10p.json"
):
    prompter = Prompter(schema_path)
    tokenizer, model = load_model(base_model, lora_weights=lora_weights)
    # data = []
    # for line in open(processed_data_path, "r"):
    #     data.append(json.loads(line))
    data = json.load(open(processed_data_path, "r"))
    # data = [item for item in data if item["value"] != "none"]
    response_list = []
    aga_num = 0

    jga_num = 0
    jga_tot = -1
    last_index_turn = ""
    last_full_state = True
    # idx_lines = open("LDST/Data/MULTIWOZ2.4_preprocess/test_5p.idx").readlines()
    for idx, sample in enumerate(tqdm(data)):
        # idx_entity = idx_lines[idx].strip().split("|||")
        # sample = {
        #     "index": idx_entity[1],
        #     "turn": idx_entity[2],
        #     "domain": idx_entity[4],
        #     "slot": idx_entity[5],
        #     "dialogue": sample["dialogue"][:sample["dialogue"].find("[domain]")],
        #     # "active": sample["active"],
        #     "value": sample["state"]
        # }
        this_index_turn = f'{sample["index"]}|{sample["turn"]}'
        if last_index_turn == "":
            last_index_turn = this_index_turn
        if last_index_turn != this_index_turn:
            last_index_turn = this_index_turn
            jga_tot += 1
            if last_full_state:
                jga_num += 1
            last_full_state = True

        if idx > 0 and idx % 100 == 0:
            print(f"AGA for 0~{idx}: {aga_num / (idx + 1)}")
            print(f"JGA for 0~{idx}: {jga_num / jga_tot}")

        noob_prompt = prompter.noob_prompt(sample["dialogue"], sample["domain"], sample["slot"])
        output = generation(noob_prompt, tokenizer, model)
        # print(output)
        response = prompter.get_response(output)

        if sample['value'].lower() == response.lower() or sample['value'].replace("not mentioned", "NONE"):
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
    print("AGA:", aga_num / len(data))
    print("JGA:", jga_num / jga_tot)
    json.dump(response_list, open(output_file, "w"))


if __name__ == "__main__":
    main()
'''
NOT NONE:
AGA for 0~3100: 0.6133505320864238
JGA for 0~3100: 0.15824915824915825
NONE:
AGA for 0~1000: 0.28471528471528473
JGA for 0~1000: 0.0
ALL:
AGA for 0~2000: 0.3803098450774613
JGA for 0~2000: 0.0
'''

'''
ALL 2.4:
AGA for 0~800: 0.630461922596754
JGA for 0~800: 0.0
NOT NONE:
AGA for 0~1000: 0.31368631368631367
JGA for 0~1000: 0.0
'''
