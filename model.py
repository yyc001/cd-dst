import os


class InferenceModel:
    def __init__(self, **kwargs):
        self.name = self.__class__

    def generate(self, prompt):
        raise NotImplementedError


def load_model(model_config: dict) -> InferenceModel:
    models = {
        "llama-2": LLaMaModel,
        "gpt-3.5-turbo": ChatGPTModel,
        "flan-t5-xxl": FlanT5Model,
        "llama-h": LlamaHModel
    }
    return models[model_config['name']](**model_config)


class LLaMaModel(InferenceModel):
    def __init__(self, model_path, tokenizer, **kwargs):
        super().__init__()
        from llama import Llama
        self.model = Llama.build(
            ckpt_dir=model_path,
            tokenizer_path=tokenizer,
            max_seq_len=2048,
            max_batch_size=4,
        )

    def generate(self, prompt):
        output = self.model.chat_completion(
            [
                [{"role": "user", "content": prompt}],
            ],
            max_gen_len=128,
            temperature=0.2,
            top_p=0.9,
        )[0]['generation']['content']
        return output


class ChatGPTModel(InferenceModel):

    def __init__(self, **kwargs):
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(
        )

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        response = response.choices[0].message.content
        return response


class FlanT5Model(InferenceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer = T5Tokenizer.from_pretrained(
            "google/flan-t5-xxl"
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl",
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate(self, input_text):
        input_ids = self.tokenizer(
            input_text,
            max_length=2048,
            return_tensors="pt"
        ).input_ids.to("cuda")

        outputs = self.model.generate(
            input_ids,
            max_length=128
        )
        output = self.tokenizer.decode(outputs[0][1:-1])
        return output


class LlamaHModel(InferenceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import torch
        import transformers
        # from transformers import LlamaTokenizer, LlamaForCausalLM
        #
        # self.tokenizer = LlamaTokenizer.from_pretrained(
        #     "NousResearch/Llama-2-7b-chat-hf"
        # )
        # self.model = LlamaForCausalLM.from_pretrained(
        #     "NousResearch/Llama-2-7b-chat-hf",
        #     device_map="auto",
        #     torch_dtype=torch.float16
        # )
        self.pipe = transformers.pipeline(
            "text-generation",
            # model="NousResearch/Llama-2-7b-chat-hf",
            model="meta-llama/Llama-2-7b-chat-hf",
            token=os.environ.get("HF_ACCESS_TOKEN"),
            max_length=2048,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, input_text):
        # input_ids = self.tokenizer(
        #     input_text,
        #     max_length=2048,
        #     return_tensors="pt"
        # ).input_ids.to("cuda")
        # outputs = self.model.generate(
        #     input_ids,
        #     max_new_tokens=128
        # )
        # output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = self.pipe(input_text)[0]['generated_text'][len(input_text):]
        return output


class ChatGLMModel(InferenceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
        self.model = model.eval()

    def generate(self, prompt):
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response


class OPTModel(InferenceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16).cuda()

        # the fast tokenizer currently does not work correctly
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)

    def generate(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        generated_ids = self.model.generate(input_ids)

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response


class LlamaAdapterModel(InferenceModel):
    def __init__(self, hf_model, adapter_path, **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import LlamaTokenizer, LlamaForCausalLM
        self.model = LlamaForCausalLM.from_pretrained(
            hf_model,
            token=os.environ.get("HF_ACCESS_TOKEN"),
            cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
            # load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        self.model.load_adapter(adapter_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            hf_model,
            cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
            token=os.environ.get("HF_ACCESS_TOKEN"),
            trust_remote_code=True
        )

    def generate(self, input_text):
        input_ids = self.tokenizer(
            input_text,
            max_length=2048,
            return_tensors="pt"
        ).input_ids.to("cuda")
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=128
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = output[len(input_text):]
        # output = self.pipe(input_text)[0]['generated_text'][len(input_text):]
        return output
