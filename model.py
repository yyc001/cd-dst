

class InferenceModel:

    def generate(self, prompt):
        raise NotImplementedError


def load_model(model_name, model_config) -> InferenceModel:
    models = {
        "llama": LLaMaModel,
        "gpt-3.5-turbo": ChatGPTModel
    }
    if model_config:
        return models[model_name](**model_config)
    else:
        return models[model_name]()


class LLaMaModel(InferenceModel):
    def __init__(self, **kwargs):
        from llama import Llama
        self.model = Llama.build(
            ckpt_dir="../llama/llama-2-7b-chat/",
            tokenizer_path="../llama/tokenizer.model",
            max_seq_len=1024,
            max_batch_size=4,
        )

    def generate(self, prompt):
        output = self.model.chat_completion(
            [
                [{"role": "user", "content": prompt}],
            ],
            max_gen_len=64,
            temperature=0.6,
            top_p=0.9,
        )[0]['generation']['content']
        return output


class ChatGPTModel(InferenceModel):

    def __init__(self, **kwargs):
        from openai import OpenAI
        self.client = OpenAI(
            api_key="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"
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
