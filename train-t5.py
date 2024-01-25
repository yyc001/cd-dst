import json
import os

import torch
# import nltk
# import evaluate
# import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
# import sentencepiece

load_dotenv(".env", verbose=True, override=True)


# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-xxl"

tokenizer = T5Tokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=os.environ.get("TRANSFORMERS_CACHE")
)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
    torch_dtype=torch.float16,
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    # target_modules=["gate_proj", "down_proj", "up_proj"],
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, peft_config)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def data_process(data_path):
    data = json.load(open(data_path))
    train_data = []
    answers = []
    for index, dialog in data.items():
        for turn in dialog:
            last_state = {k: v for k, v in turn["state"].items() if k not in turn["active_state"]}
            context = "; ".join(
                [f"{k}={v}" for k, v in last_state.items()]
            )
            if len(turn["active_state"]) > 0:
                output = f"User informed {len(turn['active_state'])} columns: " + "; ".join(
                    [f"{k}={v}" for k, v in turn["active_state"].items()]
                )
            else:
                output = "No column informed"
            text = '''Contexts: {input_context}
    Dialogue:
    {input_utterance}
    Please write the lists: (Don't write anything other than the lists themselves)'''.format(
                input_context=context,
                input_utterance=f"sys: {turn['system_utterance']} \n usr: {turn['user_utterance']}"
            )
            train_data.append(text)
            answers.append(output)
    train_data = Dataset.from_dict({"question": train_data, "answer": answers})
    return train_data


# Define the preprocessing function

def preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    # The "inputs" are the tokenized answer:
    inputs = [doc for doc in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # The "labels" are the tokenized outputs:
    labels = tokenizer(text_target=examples["answer"],
                       max_length=512,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Map the preprocessing function across our dataset
tokenized_dataset = data_process("data/MultiWOZ_2.4_processed/train.json").map(preprocess_function, batched=True)
eval_dataset = data_process("data/MultiWOZ_2.4_processed/test.json").map(preprocess_function, batched=True)

# nltk.download("punkt", quiet=True)
# metric = evaluate.load("rouge")


# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#
#     # decode preds and labels
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     # rougeLSum expects newline after each sentence
#     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
#
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#
#     return result


# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=L_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=SAVE_TOTAL_LIM,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)

trainer.train()
