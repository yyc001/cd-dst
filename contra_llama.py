# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from llama import Llama


class PromptContraDecodeLlama(Llama):

    def logits_processor(self, logits):
        print("???")
        exit(-1)
        if not self.enable_cd:
            return logits
        alpha = 0.1
        probs = torch.softmax(logits[:, -1], dim=-1)
        expert_probs = probs[::2]
        amateur_probs = probs[1::2]

        expert_probs[expert_probs < alpha * torch.max(expert_probs)] = 0
        cd_score = torch.log(expert_probs / amateur_probs)
        print(cd_score - expert_probs)
        logits[:, -1] = torch.repeat_interleave(cd_score)

        return logits
