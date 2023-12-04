import json
import os
import random
import sys
import torch
import transformers
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    # set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from prompter import DefaultPrompter


class PreprocessedDataset(Dataset):
    def __init__(self, data_path, sample=None, prompter=None, tokenize_func=None):
        self.data = json.load(open(data_path, "r"))
        if sample is not None:
            sample_num = round(len(self.data) * sample)
            random.shuffle(self.data)
            self.data = self.data[:sample_num]
        self.prompter = prompter
        self.tokenize_func = tokenize_func

    def __getitem__(self, index):
        item = self.data[index]
        if self.prompter:
            item = self.prompter.generate_prompt(item["dialogue"], item["domain"], item["slot"])
        if self.tokenize_func:
            item = self.tokenize_func(item)
        return item

    def __len__(self):
        return len(self.data)


def get_tokenizer(base_model):
    tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir="./")
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    cutoff_len = 512

    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    return tokenizer, tokenize


def get_model(base_model, resume_from_checkpoint):
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./"
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(
        model=model,
        peft_config=LoraConfig(
            bias="none",
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            # target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
        )
    )
    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model.bin"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model


def train(
        # model/data params
        # base_model: str = "NousResearch/Llama-2-7b-chat-hf",  # the only required argument
        base_model: str = "daryl149/llama-2-7b-chat-hf",
        data_path: str = "data/MultiWOZ_2.2_preprocess/train.json",
        output_dir: str = "./output_model_full",
        schema_path: str = "data/multiwoz/data/MultiWOZ_2.2/schema.json",
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    # if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    #     print(
    #         f"Training Alpaca-LoRA model with params:\n"
    #         f"base_model: {base_model}\n"
    #         f"data_path: {data_path}\n"
    #         f"output_dir: {output_dir}\n"
    #         f"batch_size: {batch_size}\n"
    #         f"micro_batch_size: {micro_batch_size}\n"
    #         f"num_epochs: {num_epochs}\n"
    #         f"learning_rate: {learning_rate}\n"
    #         f"cutoff_len: {cutoff_len}\n"
    #         f"val_set_size: {val_set_size}\n"
    #         f"lora_r: {lora_r}\n"
    #         f"lora_alpha: {lora_alpha}\n"
    #         f"lora_dropout: {lora_dropout}\n"
    #         f"lora_target_modules: {lora_target_modules}\n"
    #         f"add_eos_token: {add_eos_token}\n"
    #         f"group_by_length: {group_by_length}\n"
    #         f"wandb_project: {wandb_project}\n"
    #         f"wandb_run_name: {wandb_run_name}\n"
    #         f"wandb_watch: {wandb_watch}\n"
    #         f"wandb_log_model: {wandb_log_model}\n"
    #         f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
    #     )

    model = get_model(base_model, resume_from_checkpoint)
    tokenizer, tokenize_func = get_tokenizer(base_model)
    data = PreprocessedDataset(
        data_path,
        # sample=0.0001,
        prompter=DefaultPrompter(schema_path),
        tokenize_func=tokenize_func
    )
    # train_data, val_data = torch.utils.data.random_split(data, [len(data)-100, 100])
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs=1,
            learning_rate=1e-3,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            # evaluation_strategy="steps",
            save_strategy="steps",
            # eval_steps=200,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            # load_best_model_at_end=True,
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=True,  # faster, but produces an odd training loss curve
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()
    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

    # model.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
    # fire.Fire(train)
