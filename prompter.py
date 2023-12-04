import json
import re
import string


class DefaultPrompter:
    def __init__(self, schema_path):
        self._verbose = False
        self.prompt_v1 = '''Now you need to perform the task of multi-domain dialogue state tracking. You need to return the value of the slot I’m asking about simply based on the content of the dialogue. No explanation!
        
Input dialogue: "{text}"
        
Domain: {domain}. Slot: {slot}. it indicates {slot_explain}. {possible_values}

If the slot is not mentioned in the dialogue, just return "none".

So the value of slot <{slot}> is $$$ {value}'''
        self.prompt_v2 = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Track the state of the slot in the input dialogue. Just write a slot value with no explanation.
The slot <{slot}> means {slot_explain}. {possible_values} If the slot is not mentioned in the dialogue, just return "none".


### Input:
{text}

So the value of slot <{slot}> is $$$'''
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

        So the {slot_explain} is $$$'''
        self.prompt_v5 = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        Track the state of the slot in the input dialogue. Just write a slot value with no explanation.
        The slot <{slot}> means {slot_explain}. {possible_values}


        ### Input:
        {text}

        So the value of slot <{slot}> is $$$'''
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
            "\", \"".join(self.slot_possible_values[slot])
        ) if slot in self.slot_possible_values else ""
        return self.prompt_v1.format(
            text=text,
            domain=domain,
            slot=slot,
            slot_explain=self.slot_explain[slot],
            possible_values="",
            value=value
        )

    def get_response(self, text):
        response = text.split("$$$")[1].strip().lower()
        if self._verbose:
            return response
        translator = str.maketrans('', '', string.punctuation)
        response = response.translate(translator)
        return response
