import json
import os
import tiktoken
from tqdm import tqdm
from prompter import Prompter


def sys_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


def main(
        schema_path="data/multiwoz/data/MultiWOZ_2.2/schema.json",
        processed_data_path="data/MultiWOZ_2.2_preprocess/test.json"
):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    prompter = Prompter(schema_path)
    data = json.load(open(processed_data_path, "r"))
    # data = [item for item in data if item["value"] != "none"]
    print("data length:", len(data))
    total_token = 0
    for idx, sample in enumerate(tqdm(data)):
        # prompt = prompter.generate_prompt(sample["dialogue"], sample["domain"], sample["slot"])
        prompt = prompter.generate_prompt(sample["dialogue"], sample["domain"], sample["slot"])
        total_token += len(encoding.encode(prompt))
    print("total input token:", total_token)  # 47570187 $47.570187
    # output ???


if __name__ == "__main__":
    main()
