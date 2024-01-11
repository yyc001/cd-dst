import json
import os

import openai
from tqdm import tqdm
from prompter import Prompter, DatabasedPrompter
from openai import OpenAI
from dotenv import load_dotenv


def main(
        schema_path="data/multiwoz/data/MultiWOZ_2.2/schema.json",
        processed_data_path="data/MultiWOZ_2.4_preprocess/test.json",
        output_file="data/MultiWOZ_2.4_preprocess/test_out_gpt.json",
):
    load_dotenv()
    client = OpenAI()
    prompter = DatabasedPrompter("data/MULTIWOZ2.4", schema_path)
    data = json.load(open(processed_data_path, "r"))
    # data = [item for item in data if item["value"] != "none"]
    if not os.path.isfile(output_file):  #
        result_out = open(output_file, "w", encoding='utf-8')
        begin_id = 0
        print("——————————————————————————————Write from scratch——————————————————————————————")
    else:  #
        with open(output_file, "r") as f:
            lines = f.readlines()
            begin_id = len(lines)
            f.close()
        print(f"——————————————————————————————Write from line {begin_id}——————————————————————————————")
        result_out = open(output_file, "a", encoding='utf-8')
    tot = 0
    acc = 0
    for idx in tqdm(range(begin_id, len(data))):
        sample = data[idx]
        # if sample['value'] == "not mentioned" or sample["value"] == "none":
        #     continue
        tot += 1
        # input_text = prompter.get_input_text(sample["dialogue"], sample["domain"], sample["slot"])
        messages = prompter.generate_message_history(sample["dialogue"], sample["domain"], sample["slot"])
        # messages = [
        #         # {"role": "system", "content": prompter.instruction},
        #         # {"role": "user", "content": input_text},
        #         {"role": "user", "content": prompt},
        #     ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        # response = openai.ChatCompletion.create(
        #     engine="gpt-3.5-turbo",
        #     prompt={"messages": prompt, "instruction": prompter.instruction},
        #     max_tokens=150  # You can adjust the max_tokens based on the desired response length
        # )
        response = response.choices[0].message.content

        # print(output)
        response = prompter.get_response(response)
        # print(sample['value'], response)
        if sample['value'].lower() == response.lower():
            acc += 1
        else:
            # print(sample["dialogue"])
            print(messages[0]["content"])
            print(f"{acc / tot}|||{sample['domain']}-{sample['slot']}|||{sample['value']}|||{response}")

        response_entity = {
            "index": sample["index"],
            "turn": sample["turn"],
            "domain": sample["domain"],
            "slot": sample["slot"],
            # "active": sample["active"],
            "value": response,
            "ground_truth": sample["value"]
        }
        result_out.write(json.dumps(response_entity) + "\n")
        result_out.flush()
    # json.dump(response_list, open(output_file, "w"))


if __name__ == "__main__":
    main()
