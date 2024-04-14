import json
import os
from dotenv import load_dotenv
load_dotenv(".env", verbose=True, override=True)

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def data_process(data_path):
    data = json.load(open(data_path))
    train_data = []
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
Please write the lists: (Don't write anything other than the lists themselves)
{output} </s>'''.format(
                input_context=context,
                input_utterance=f"sys: {turn['system_utterance']} \n usr: {turn['user_utterance']}",
                output=output
            )
            train_data.append(text)
    train_data = Dataset.from_dict({"text": train_data})
    print(train_data[233])
    return train_data


def train(model_name, data_path, output_dir, eval_path, **kwargs):
    resume = os.path.exists(output_dir)
    # print(os.environ.get("HF_ACCESS_TOKEN"))
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
        # print(torch.cuda.current_device(), "-------------", local_rank)
        # exit(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(range(local_rank, local_rank+2))
        device_map = {"":local_rank}
    model = AutoModelForCausalLM.from_pretrained(
    # model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        # cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
        # quantization_config=BitsAndBytesConfig(
        #     load_in_8bit=True,
        # ),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
        token=os.environ.get("HF_ACCESS_TOKEN"),
        trust_remote_code=True
    )
    # print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
    # print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    # exit(-1)
    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = "left"

    # import re
    # pattern = r'\((\w+)\): Linear'
    # linear_layers = re.findall(pattern, str(model.modules))
    # target_modules = list(set(linear_layers))
    # print(target_modules)
    # exit(-1)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["gate_proj", "down_proj", "up_proj"],
        # target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")
    # model = get_peft_model(model, peft_config)
    # print("!")

    model.enable_input_require_grads()
    training_arguments = TrainingArguments(
        # 1. 常规参数
        output_dir=output_dir,  # 结果/检查点输出路径
        per_device_train_batch_size=4,  # 单卡batchsize
        optim="adamw_torch",  # 优化器名称
        learning_rate=1e-3,  # 学习率
        eval_steps=100,  # 多少step进行一次评估
        save_steps=100,  # 多少step进行一次检查点保存
        logging_steps=100,  # 多少step记录一次训练loss
        evaluation_strategy="steps",
        group_by_length=False,
        # max_steps=max_steps, # 最大训练steps 和 num_train_epochs 二选一
        num_train_epochs=10,  # 最大训练 epoch
        # 2. 节省显存参数
        gradient_accumulation_steps=4,  # 梯度累计
        # gradient_checkpointing=True,  # 梯度检查点
        # max_grad_norm=0.3,
        # 3. 类型参数
        # fp16=True,
        bf16=True,
        # 4. 学习率调节
        lr_scheduler_type="cosine",
        # warmup_ratio=warmup_ratio,
        warmup_steps=100,

        # ...
        # dataloader_num_workers=4,
        # dataloader_pin_memory=True
        ddp_find_unused_parameters=False if ddp else None
    )

    eval_data = data_process(eval_path)
    train_data = data_process(data_path)
    # print("?")
    class MyTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # print(inputs)
            # print(tokenizer.batch_decode(inputs.input_ids))
            # inputs["labels"]=None
            # while True:
            #     outputs = model(**inputs)
            
            
            outputs = model(**inputs)
            # print(outputs)
            # exit(0)
            loss = outputs["loss"]
            print(loss)
            return (loss, outputs) if return_outputs else loss
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=256,  # 序列的最大长度
        tokenizer=tokenizer,
        args=training_arguments,
    )
    # 开启模型训练
    # print("BG")
    trainer.train(resume_from_checkpoint=resume)
    # 最终结果保存
    trainer.model.save_pretrained(output_dir)
    trainer.save_model()


if __name__ == "__main__":
    # notebook_laucher
    train(
        model_name="meta-llama/Llama-2-7b-hf",
        # model_name="mistralai/Mistral-7B-Instruct-v0.2",
        data_path="data/MultiWOZ_2.4_processed/train.json",
        eval_path="data/MultiWOZ_2.4_processed/test.json",
        output_dir="checkpoints/Llama-2-7b-hf/"
        # output_dir="checkpoints-Mistral-7B-Instruct-v0.2-1/"
    )
