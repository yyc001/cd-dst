import copy
from types import MethodType

if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5Tokenizer, LogitsProcessorList, MinLengthLogitsProcessor, \
    GenerationMixin

    expert = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    amateur = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions
    import torch


    def forward(self, **kwargs):
        print(kwargs.keys())
        print(kwargs["decoder_input_ids"])
        return self.expert_forward(**kwargs)
        expert_out = self.expert_forward(**kwargs)
        kwargs = copy.deepcopy(kwargs)
        amateur_out = amateur.forward(**kwargs)  # (8x1024 and 512x384) (8x768 and 512x384) (8x1024 and 768x768)
        # print(expert_out.logits.shape)
        alpha = 0.1
        amateur_probs = torch.softmax(amateur_out.logits[:, -1], dim=-1)
        expert_probs = torch.softmax(expert_out.logits[:, -1], dim=-1)
        # print(expert_probs.shape, amateur_probs.shape)
        # expert_probs = torch.tensor([[123, 12, 456], [55, 66, 1]])
        # print(expert_probs)
        expert_probs[expert_probs > alpha * expert_probs.max(dim=1, keepdim=True).values] = 0
        # print(expert_probs)
        cd_score = torch.log(expert_probs / amateur_probs)
        # print(torch.max(cd_score, dim=1).values, torch.min(cd_score, dim=1).values)
        expert_out.logits = torch.unsqueeze(cd_score, dim=1)
        return expert_out


    expert.expert_forward = MethodType(expert.forward, expert)
    expert.forward = MethodType(forward, expert)
    # expert.amateur = amateur

    text = "hello hi hello hi hello hi hello"
    tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(tokenized.input_ids)
    outputs = expert.generate(tokenized.input_ids)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(response)
    expert.generate(
        tokenized.input_ids,
        generation_config=None,
        assistant_model=amateur
    )

    # encoder_input_ids = tokenized
    # input_ids = torch.ones((3, 1), device=expert.device, dtype=torch.long)
    # input_ids = input_ids * expert.config.decoder_start_token_id
    # print(expert.config.bos_token_id)
    # model_inputs = expert.prepare_inputs_for_generation(input_ids=expert.config.bos_token_id, encoder_outputs=encoder_outputs)
    # expert_output = expert(
    #     **expert.prepare_inputs_for_generation(**tokenized),
    #     return_dict=True,
    #     output_attentions=False,
    #     output_hidden_states=False
    # )
    # print(expert_output)


class Gen(GenerationMixin):
    def _reorder_cache(self, past_key_values, beam_idx):
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

'''
>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
>>> # lets run beam search using 3 beams
>>> num_beams = 3
>>> # define decoder start token ids
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id
>>> # add encoder_outputs to model keyword arguments
>>> model_kwargs = {
...     "encoder_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }
>>> # instantiate beam scorer
>>> beam_scorer = BeamSearchScorer(
...     batch_size=1,
...     num_beams=num_beams,
...     device=model.device,
... )
>>> # instantiate logits processors
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
...     ]
... )
>>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt bist du?']
'''