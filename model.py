

class InferenceModel:
    def __init__(self):
        self.name = self.__class__

    def generate(self, prompt):
        raise NotImplementedError


def load_model(model_config: dict) -> InferenceModel:
    models = {
        "llama-2": LLaMaModel,
        "gpt-3.5-turbo": ChatGPTModel,
        "flan-t5-xxl": FlanT5Model
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
    def __init__(self):
        super().__init__()
        from transformers import pipeline
        self.pipe = pipeline("text2text-generation", model="google/flan-t5-xxl")

    def generate(self, prompt):
        response = self.pipe(prompt
                             # max_length=30,
                             # num_return_sequences=2,
                             )
        return response[0]['generated_text']
