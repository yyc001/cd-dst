import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

from evaluate import Evaluator
from model import load_model
from prompter import SingleReturnPrompter


def sys_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


def main(
        model_name="",
        model_config=None,
        processed_data_path="",
        output_file="",
        resume=True
):
    prompter = SingleReturnPrompter()
    generator = load_model(model_name, model_config)
    data = json.load(open(processed_data_path, "r"))
    evaluator = Evaluator()

    logging.info(f"Model: {model_name}, {model_config}")
    logging.info("Input data: " + processed_data_path)
    logging.info("Prompt example: \n" + prompter.generate_prompt("$1", "$2", {"$3": "$4"}))

    if os.path.exists(output_file) and resume:
        predicted = json.load(open(output_file, "r"))
    else:
        predicted = {}

    for index, dialog in tqdm(data.items()):

        if index in predicted:
            logging.warning("Skip index: " + index)
            continue

        predicted_states = []
        for turn in dialog:
            prompt = prompter.generate_prompt(
                system_uttr=turn["system_utterance"],
                user_uttr=turn["user_utterance"],
                last_state={k: v for k, v in turn["state"].items() if k not in turn["active_state"]}
            )
            logging.root.setLevel(logging.ERROR)
            output = generator.generate(prompt)
            logging.root.setLevel(LOGGING_LEVEL)
            active_state = prompter.get_response(output)
            predicted_states.append({
                "turn_id": turn["turn_id"],
                "output": output,
                "active_state": active_state,
            })
            logging.debug(f"output: {output}")
            logging.debug(f"predicted: {active_state}")
            logging.debug(f"ground truth: {turn['active_state']}")
        predicted[index] = predicted_states

        # logging.info("\n".join(
        #         [f"{k}: {v}" for k, v in evaluator.evaluate(predicted, data, pred_only=True).items()]
        #     ))
        json.dump(predicted, open(output_file, "w"))
    print("\n".join(
            [f"{k}: {v}" for k, v in evaluator.evaluate(predicted, data, pred_only=True).items()]
        ))


if __name__ == "__main__":
    LOGGING_LEVEL = logging.INFO
    logging.root.setLevel(LOGGING_LEVEL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="data/MultiWOZ_2.4_processed/test_out.json")
    parser.add_argument('--processed_data_path', type=str, default="data/MultiWOZ_2.4_processed/test.json")
    parser.add_argument('--model_name', type=str, default="llama-2-7b-chat")
    parser.add_argument('--resume', type=bool, default=True)
    args = parser.parse_args()
    main(**args.__dict__)
