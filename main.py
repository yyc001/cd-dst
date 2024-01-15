import argparse
import json
import logging
import os
import torch

from tqdm import tqdm

from evaluate import Evaluator
from dotenv import load_dotenv
from model import load_model
from prompter import SingleReturnPrompter


def inference(
        model_config,
        data_path,
        output_file,
        resume,
        **kwargs
):
    prompter = SingleReturnPrompter()
    model_config = json.load(open(model_config))
    generator = load_model(model_config)
    data = json.load(open(data_path, "r"))

    logging.info(f"Model: {model_config}")
    logging.info("Input data: " + data_path)
    # logging.info("Prompt example: \n" + prompter.generate_prompt("$1", "$2", {"$3": "$4"}))
    logging.warning("--- test model --- ")
    logging.warning(generator.generate("Hello, what is your name?"))
    logging.warning("--- test model end --- ")

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
            with torch.no_grad():
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
        # evaluate_process(data_path, output_file, False, **kwargs)
        json.dump(predicted, open(output_file, "w"), indent=2)

    evaluate_process(data_path, output_file, False)


def evaluate_process(data_path, output_file, reparse, **kwargs):
    data = json.load(open(data_path, "r"))
    evaluator = Evaluator()
    predicted = json.load(open(output_file, "r"))
    if reparse:
        predicted_new = predicted
        prompter = SingleReturnPrompter()
        for index, dialog in tqdm(predicted.items()):
            for i in range(len(dialog)):
                predicted_new[index][i]['active_state'] = prompter.get_response(dialog[i]['output'])
        predicted = predicted_new
        json.dump(predicted, open(output_file, "w"), indent=2)

    print("\n".join(
        [f"{k}: {v}" for k, v in evaluator.evaluate(predicted, data, pred_only=True).items()]
    ))


if __name__ == "__main__":
    load_dotenv()
    LOGGING_LEVEL = logging.INFO
    logging.root.setLevel(LOGGING_LEVEL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, choices=["inference", "evaluation"])
    parser.add_argument('--model_config', type=str, default="")
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reparse', action='store_true')
    args = parser.parse_args()
    print("\n".join(
        [f"{k}: {v}" for k, v in args.__dict__.items()]
    ))

    if args.job == "inference":
        inference(**args.__dict__)
    elif args.job == "evaluation":
        evaluate_process(**args.__dict__)
