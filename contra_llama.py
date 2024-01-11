# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from llama import Llama


class PromptContraDecodeLlama(Llama):

    def logits_processor(self, logits):
        if not hasattr(self, "enable_cd") or not self.enable_cd:
            return logits
        # print(logits.shape)
        alpha = 0.1
        probs = torch.softmax(logits[:, -1], dim=-1).cuda()
        # print(probs.shape)
        expert_probs = probs[::2]
        amateur_probs = probs[1::2]
        # print(expert_probs.shape, amateur_probs.shape)

        expert_probs[expert_probs < alpha * torch.max(expert_probs)] = 0
        cd_score = torch.log(expert_probs / amateur_probs)
        # print(cd_score.shape)
        probs_cd = torch.cat([cd_score, cd_score], dim=0)
        # print(probs_cd.shape)
        logits[:, -1] = probs_cd
        # print(torch.max(cd_score), torch.argmax(cd_score))
        return logits
