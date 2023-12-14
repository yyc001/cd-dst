import json
import re
import string
import os.path as osp
from typing import Union


class DefaultPrompter:
    def __init__(self, schema_path):
        self.prompt_v1 = '''Now you need to perform the task of multi-domain dialogue state tracking. You need to return the value of the slot I’m asking about simply based on the content of the dialogue. No explanation!
        
Input dialogue: "{text}"
        
Domain: {domain}. Slot: {slot}. it indicates {slot_explain}. {possible_values}

If the slot is not mentioned in the dialogue, just return "none".

So the value of slot <{slot}> is $$$ {value}'''
        self.prompt_v2 = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Track the state of the slot in the input dialogue.
The slot <{slot}> means {slot_explain}. {possible_values}
Just return a value of the slot. No explanation! No need for a complete sentence!
If the slot is not mentioned in the dialogue, just return "NONE"!


### Input:
{text}

So the value of slot <{slot}> is ### Response:{value}'''
        self.prompt_v3 = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Track the state of the slot in the input dialogue. Just write a slot value with no explanation.

### Input:
{text}

So the value of slot <{slot}> is $$$'''
        self.prompt_v4 = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        Track the state of the slot in the input dialogue. Just write a value with no explanation.

        ### Input:
        {text}

        So the {slot_explain} is $$${value}'''
        self.prompt_v5 = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Track the state of the slot in the input dialogue. Just write a slot value with no explanation.
The slot is {slot}. It indicates {slot_explain}. {possible_values}
If the slot is not mentioned in the dialogue, just return "none".


### Input:
{text}

### Response:
So the value of slot is {value}'''
        self.prompt_v6 = '''{text}

        So the value of {slot_explain} is $$$ {value}'''
        self.prompt_v7 = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Track the state of the slot in the input dialogue.
The slot is {slot}. It indicates {slot_explain}. {possible_values}
Return the value of the slot without any explanation, no need for a sentence or short phrase.
If the slot is not mentioned in the dialogue, just return "NONE"!

### Input:
{text}

### Example response:
NONE

### Response:
{slot} = {value}'''
        self.possible_values_prompt = "This slot is categorical and you can only choose from the following available values: \"{}\""
        self.slot_explain = {}
        self.slot_possible_values = {}
        schema = json.load(open(schema_path, "r"))
        for doamin_entity in schema:
            for slot_entity in doamin_entity["slots"]:
                slot = slot_entity["name"]
                self.slot_explain[slot] = slot_entity["description"]
                if slot_entity["is_categorical"] is True:
                    self.slot_possible_values[slot] = slot_entity["possible_values"]

    def generate_prompt(self, text, domain, slot, value=""):
        possible_values = self.possible_values_prompt.format(
            "\", \"".join(["none"] + self.slot_possible_values[slot])
        ) if slot in self.slot_possible_values else ""
        return self.prompt_v5.format(
            text=text,
            domain=domain,
            slot=slot,
            slot_explain=self.slot_explain[slot],
            possible_values=possible_values,
            value=value
        )

    def get_response(self, text):
        response = text.split("So the value of slot is ")[1].strip().lower()
        response = response.replace("</s>", "").replace(".", "").replace("\n", "").replace("\"", "")
        # translator = str.maketrans('', '', string.punctuation)
        # response = response.translate(translator)
        if len(response) == 0:
            response = "none"
        return response


class Prompter:
    # __slots__ = ("template", "_verbose")

    def __init__(self, schema_path):
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
            "response_split": "### Response:"
        }
        self.noob_template = {
            "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
            "response_split": "### Response:"
        }
        self.instruction = "Now you need to perform the task of multi-domain dialogue state tracking. You need to return the value of the slot I’m asking about simply based on the content of the dialogue. No explanation!"
        self.possible_values_prompt = "This slot is categorical and you can only choose from the following available values: {}."
        self.slot_explain = {}
        self.slot_possible_values = {}
        schema = json.load(open(schema_path, "r"))
        for domain_entity in schema:
            for slot_entity in domain_entity["slots"]:
                slot = slot_entity["name"]
                self.slot_explain[slot] = slot_entity["description"]
                if slot_entity["is_categorical"] is True:
                    self.slot_possible_values[slot] = slot_entity["possible_values"]

    def get_input_text(self, text, domain, slot):
        if "-" in slot:
            slot = slot.split("-")[1]
        intput_template = "Input dialogue: {text} [domain] {domain} [slot] {slot}, it indicates {slot_explain} {possible_values} If the slot is not mentioned in the dialogue, just return NONE.\n So the value of slot <{domain_slot}> is \n"
        intput_template2 = "Input dialogue: {text} So the value of slot <{domain_slot}> is \n"
        possible_values = self.possible_values_prompt.format(
            ", ".join(self.slot_possible_values[f"{domain}-{slot}"])
        ) if f"{domain}-{slot}" in self.slot_possible_values else ""
        input_text = intput_template.format(
            text=text,
            domain=domain,
            slot=slot,
            slot_explain=self.slot_explain[f"{domain}-{slot}"],
            possible_values=possible_values,
            domain_slot=f"{domain}-{slot}"
        )
        return input_text

    def generate_prompt(self, text, domain, slot, value="", pair=False):
        if "-" in slot:
            slot = slot.split("-")[1]
        slot = slot.lower()
        res = self.template["prompt_input"].format(
            instruction=instruction,
            input=self.get_input_text(text, domain, slot),
            output=value
        )
        if pair:
            noob_res = self.template["prompt_input"].format(
                instruction="Track the state of the slot in the input dialogue.",
                input="{text} \n So the value of slot <{domain_slot}> is \n".format(
                    text=text,
                    domain_slot=f"{domain}-{slot}"
                ),
                output=value
            )
            return res, noob_res
        return res

    def dummy_prompt(
            self,
            instruction: str,
            input=None,
            label=None
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, output=label
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction, output=label
            )
        # if self._verbose:
        #     print(res)
        return res

    def noob_prompt(self, text, domain, slot, value=""):
        if "-" in slot:
            slot = slot.split("-")[1]
        noob_res = self.template["prompt_input"].format(
            instruction="Track the state of the slot in the input dialogue.",
            input=" {text} \n  \n So the value of slot <{domain_slot}> is \n".format(
                text=text,
                domain_slot=f"{domain}-{slot}"
            ),
            output=value
        )
        return noob_res

    def get_response(self, response: str) -> str:
        if self.template["response_split"] in response:
            response = response.split(self.template["response_split"])[1].strip()
        if " is:" in response:
            response = response.split(" is:")[1]
        if " is " in response:
            response = response.split(" is ")[1]
        response = response.split(",")[0].split(".")[0].replace("</s>", "").replace("<s>", "").replace("\"", "").strip()
        return response


class LDSTPrompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
            "response_split": "### Response:"
        }

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = "",
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, output=label
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction, output=label
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, response: str) -> str:
        if self.template["response_split"] in response:
            response = response.split(self.template["response_split"])[1].strip()
        if " is:" in response:
            response = response.split(" is:")[1]
        if " is " in response:
            response = response.split(" is ")[1]
        response = response.split(",")[0].split(".")[0].replace("</s>", "").replace("<s>", "").replace("\"", "").strip()
        return response
