import os
from types import MethodType

from transformers import LogitsProcessorList


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
        "llama-h": LlamaHModel,
        "tuned-llama": LlamaAdapterModel,
        "tuned-t5": T5AdapterModel,
        "llama-2-contra": PromptContraLLaMaModel,
        "contra-decode": ContraModel,
        "Mistral-7B": MistralModel,
        "SCDModel": SCDModel,
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


class PromptContraLLaMaModel(InferenceModel):
    def __init__(self, model_path, tokenizer, **kwargs):
        super().__init__()
        from contra_llama import PromptContraDecodeLlama
        self.model = PromptContraDecodeLlama.build(
            ckpt_dir=model_path,
            tokenizer_path=tokenizer,
            max_seq_len=2048,
            max_batch_size=4,
        )

    def generate(self, prompt):
        output = self.model.chat_completion(
            [
                [{"role": "user", "content": prompt}],
                [{"role": "user", "content": prompt[prompt.find("Contexts:"):]}],
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
            # cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
            # load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        
        self.model.load_adapter(adapter_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            hf_model,
            # cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
            token=os.environ.get("HF_ACCESS_TOKEN"),
            trust_remote_code=True
        )
        self.tokenizer.pad_token_id = 0 
        self.tokenizer.padding_side = "left"

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
        # print(output)
        output = output[len(input_text):]
        # output = self.pipe(input_text)[0]['generated_text'][len(input_text):]
        return output


class T5AdapterModel(InferenceModel):
    def __init__(self, hf_model, adapter_path, **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from peft import PeftModel
        model = T5ForConditionalGeneration.from_pretrained(
            hf_model,
            # cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
            # load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            # load_in_8bit=True,
        )
        self.model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map="auto"
        )
        self.model.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(
            hf_model,
            # cache_dir=os.environ.get("TRANSFORMERS_CACHE")
        )

    def generate(self, input_text):
        input_ids = self.tokenizer(
            input_text,
            max_length=2048,
            return_tensors="pt"
        ).input_ids.to("cuda")

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output


def load_config(hf_model, adapter_path, model_class):
    import torch
    model = model_class.from_pretrained(
        hf_model,
        # cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
        token=os.environ.get("HF_ACCESS_TOKEN"),
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        # load_in_8bit=True
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map="auto"
        )
    return model


class PromptContraModel(InferenceModel):
    def __init__(self, hf_model, adapter_path, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoTokenizer
        self.model = load_config(hf_model, adapter_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model,
            cache_dir=os.environ.get("TRANSFORMERS_CACHE")
        )

    def generate(self, input_text: str):
        input_ids = self.tokenizer(
            [
                input_text,
                input_text[input_text.find("Contexts:"):]
            ],
            max_length=2048,
            return_tensors="pt"
        ).input_ids.to("cuda")

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            logits_processor=LogitsProcessorList(

            )
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output


class ContraModel(InferenceModel):
    def __init__(self,
                 expert_model, expert_adapter, amateur_model, amateur_adapter,
                 amateur_scale, expert_logc, model_class, **kwargs):
        super().__init__(**kwargs)
        import transformers
        from transformers import AutoTokenizer, AutoModel
        model_class = eval(model_class) if model_class else AutoModel
        self.expert = load_config(expert_model, expert_adapter, model_class)
        self.amateur = load_config(amateur_model, amateur_adapter, model_class)
        self.tokenizer = AutoTokenizer.from_pretrained(
            expert_model,
            # cache_dir=os.environ.get("TRANSFORMERS_CACHE")
        )
        self.amateur_scale = amateur_scale
        self.expert_logc = expert_logc

    def prepare_inputs_and_kwargs(self, model, input_ids):
        model_kwargs = {}
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            input_ids, self.tokenizer.bos_token_id, model_kwargs
        )
        if model.config.is_encoder_decoder:
            model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
            input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
                batch_size=inputs_tensor.shape[0],
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
            )
        return input_ids, model_kwargs

    def generate(self, input_text, max_length=128):
        import torch
        inputs = self.tokenizer(
            input_text,
            max_length=2048,
            return_tensors="pt"
        ).input_ids
        input_ids, model_kwargs_expert = self.prepare_inputs_and_kwargs(self.expert, inputs)
        input_ids2, model_kwargs_amateur = self.prepare_inputs_and_kwargs(self.amateur, inputs)
        # print(input_ids, input_ids2)
        assert input_ids.equal(input_ids2)
        with torch.no_grad():
            while input_ids.shape[-1] < max_length:
                model_inputs_expert = self.expert.prepare_inputs_for_generation(input_ids, **model_kwargs_expert)
                outputs_expert = self.expert(
                    **model_inputs_expert,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                model_inputs_amateur = self.amateur.prepare_inputs_for_generation(input_ids, **model_kwargs_amateur)
                outputs_amateur = self.expert(
                    **model_inputs_amateur,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                logits_expert = outputs_expert.logits[:, -1, :]
                logits_amateur = outputs_amateur.logits[:, -1, :]
                logp_expert = logits_expert.log_softmax(dim=-1)
                logp_amateur = logits_amateur.log_softmax(dim=-1)
                next_tokens_scores = logp_expert - self.amateur_scale * logp_amateur
                # print(logp_expert.shape)
                # print(torch.max(logp_expert, dim=-1))
                # exit(0)
                next_tokens_scores[
                    logp_expert < torch.max(logp_expert, dim=-1).values + self.expert_logc
                    ] = float("-inf")

                next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs_expert = self.expert._update_model_kwargs_for_generation(
                    outputs_expert, model_kwargs_expert, is_encoder_decoder=self.expert.config.is_encoder_decoder
                )
                model_kwargs_amateur = self.amateur._update_model_kwargs_for_generation(
                    outputs_amateur, model_kwargs_amateur, is_encoder_decoder=self.amateur.config.is_encoder_decoder
                )

                if next_tokens[0] == self.tokenizer.eos_token_id:
                    break

        output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output


class MistralModel(InferenceModel):
    def __init__(self, hf_model, adapter_path, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model = load_config(hf_model, adapter_path, AutoModelForCausalLM)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, input_text):
        input_ids = self.tokenizer(
            input_text,
            max_length=2048,
            return_tensors="pt"
        )
        outputs = self.model.generate(
            input_ids.input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128
        )
        output = self.tokenizer.decode(outputs[0])
        print("------------------------------------------")
        print(output)
        output = output[len(input_text):]
        return output


class SCDModel(InferenceModel):
    def __init__(self,
                 hf_model, adapter_path, model_class, **kwargs):
        super().__init__(**kwargs)
        import transformers
        from transformers import AutoTokenizer, AutoModel
        model_class = eval(model_class) if model_class else AutoModel
        self.model = load_config(hf_model, adapter_path, model_class)
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model,
            # cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
            token=os.environ.get("HF_ACCESS_TOKEN"),
            trust_remote_code=True
        )
        self.tokenizer.pad_token_id = 0 
        self.tokenizer.padding_side = "left"
    
    def generate(self, input_text, max_length=128):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        class MyLogitsProcessor(LogitsProcessor):

            def __init__(self, outer, prompt_len):
                from synchromesh import LarkCompletionEngine, StreamingCSD
                self.outer = outer
                self.prompt_len = prompt_len
                vocab = [v for k, v in
                        sorted([(t_id, self.outer.tokenizer.decode([t_id]))
                                for _, t_id in self.outer.tokenizer.get_vocab().items()])]

                # HACK: Is there a better way to know if a token has a prefix space?
                # We should only need this for LlamaTokenizer
                # (as it's the most popular SentencePiece derivative right now - others would need this too).
                # if 'Llama' in self.outer.tokenizer.__class__.__name__:
                for i in range(len(vocab)):
                    t = vocab[i]
                    if 2 * len(t) != len(self.outer.tokenizer.decode([i, i], add_special_tokens=False)):
                        vocab[i] = ' ' + t
                    if t == '':
                        vocab[i] = ' '

                completion_engine = LarkCompletionEngine(open("grammar.lark").read(), 'start' if self.outer.model.config.is_encoder_decoder else 'begin', False)
                self.constraint_stream = StreamingCSD(completion_engine, vocab, False)

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                # current_str = self.outer.tokenizer.decode(input_ids[0])
                # print(current_str, "**********")
                if input_ids.shape[-1] > self.prompt_len:
                    self.constraint_stream.feed_prediction(input_ids[0][-1])

                next_token = torch.argmax(scores, dim=-1)

                # if input_ids.shape[-1] == 1 and next_token in [6674, 465]:
                #     return scores

                if next_token in [self.outer.tokenizer.eos_token_id, self.outer.tokenizer.bos_token_id, 29871]:
                    return scores
                    
                if self.constraint_stream.can_token_follow(next_token):
                    return scores

                current_str = self.outer.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=False)
                print(f'"{current_str}" + "{self.outer.tokenizer.decode(next_token)}"', next_token, "|", self.constraint_stream.get_current_prediction())
                
                valid_tokens = self.constraint_stream.get_valid_tokens()
                if not valid_tokens:
                                        # print("EOS")
                    # scores[0][self.outer.tokenizer.eos_token_id] = 1
                    return scores
                
                print("------")
                valid_tokens_mask = torch.zeros(scores.shape[-1], dtype=torch.bool)
                valid_tokens_mask[valid_tokens] = True
                # if None in valid_tokens_set:
                #     valid_tokens_set.remove(None)
                scores[0][~valid_tokens_mask] = float('-inf')

                next_token = torch.argmax(scores, dim=-1)
                next_str = self.outer.tokenizer.decode(torch.cat([input_ids[0][self.prompt_len:], next_token], dim=-1), skip_special_tokens=False)
                print(next_str)
                print("******")

                return scores
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        output = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128,
            logits_processor=LogitsProcessorList([MyLogitsProcessor(self, 1 if self.model.config.is_encoder_decoder else input_ids.shape[-1])])
        )
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # print(output)
        return output
