from model import ContraModel
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel


# model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
# inputs_ids = tokenizer("hello, ", return_tensors="pt").input_ids
# inputs_ids = model.generate(inputs_ids)
# generated = tokenizer.batch_decode(inputs_ids)
# print(generated)

model = ContraModel(
    "openai-community/gpt2-large", None,
    "openai-community/gpt2", None,
    0.0, -2.3,
    "transformers.GPT2LMHeadModel"
)

text = "hello, "
generated = model.generate("hello, ")
print(generated)

