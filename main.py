import os
import subprocess

from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, pipeline
import torch
import sys


def sys_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


def load_model():
    # base_model = "decapoda-research/llama-7B-hf"
    base_model = "daryl149/llama-2-7b-chat-hf"
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        # local_files_only=True
    )
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    tokenizer = LlamaTokenizer.from_pretrained(base_model,
                                               # local_files_only=True
                                               )
    # tokenizer.bos_token_id = 1
    # tokenizer.eos_token_id = 2
    # tokenizer.pad_token_id = 0
    return tokenizer, model


def generate_response(prompt, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                num_beams=1
            ),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
        )
    # print(generation_output)
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    response = output.split("### Response:")[1].strip()
    return response


exp_prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Track the state of the slot in the input dialogue. Just write a slot value with no explanation as short as possible.

### Input:
 [USER] I want to find a moderately priced restaurant.  [SYSTEM] I have many options available for you! Is there a certain area or cuisine that interests you? [USER] Yes I would like the restaurant to be located in the center of the attractions.  [SYSTEM] There are 21 restaurants available in the centre of town. How about a specific type of cuisine? [USER] i need to know the food type and postcode and it should also have mutliple sports [SYSTEM] I am sorry I do not understand what you just said. Please repeat in a way that makes sense.  

So the value of slot <restaurant-pricerange> is 



### Response:
'''


def main():
    sys_proxy()
    '''
    pipe = pipeline("text-generation",
                    model="daryl149/llama-2-7b-chat-hf",
                    device_map="auto",
                    max_new_tokens=128
                    )
    print("pipeline")
    output = pipe([exp_prompt])
    output = output[0][0]["generated_text"].split("### Response:")[1].strip()
    print("output:", output)
    '''

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    tokenizer, model = load_model()

    prompt = r"Below is an instruction that describes a task, paired with an input that provides further context. Write a " \
             r"response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{" \
             r"input}\n\n### Response:"
    prompt = prompt.format(
        instruction="Track the state of the slot <attraction-area> in the input dialogue.",
        input=r"[USER] I want to find a moderately priced restaurant.  [SYSTEM] I have many options available for you! Is "
              r"there a certain area or cuisine that interests you? \n So the value of slot <attraction-area> is")
    # print(prompt)
    print("response:", generate_response(prompt, tokenizer, model))

    print("response:", generate_response(exp_prompt, tokenizer, model))


if __name__ == "__main__":
    main()
