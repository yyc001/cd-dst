import glob
import json
import os
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
        self.domain_explain = {}
        self.slot_explain = {}
        self.slot_possible_values = {}
        schema = json.load(open(schema_path, "r"))
        for domain_entity in schema:
            self.domain_explain[domain_entity["service_name"]] = domain_entity["description"]
            for slot_entity in domain_entity["slots"]:
                slot = slot_entity["name"]
                self.slot_explain[slot] = slot_entity["description"]
                if slot_entity["is_categorical"] is True:
                    self.slot_possible_values[slot] = slot_entity["possible_values"]

    def get_input_text(self, text, domain, slot):
        if "-" in slot:
            slot = slot.split("-")[1]
        intput_template = "Input dialogue: {text} [domain] {domain}, it indicates {domain_explain}. [slot] {slot}, it indicates {slot_explain}. {possible_values} If the slot is not mentioned in the dialogue, just return NONE.\n So the value of slot <{domain_slot}> is \n"
        intput_template2 = "Input dialogue: {text} So the value of slot <{domain_slot}> is \n"
        input_template_gpt = "Perform the task of multi-domain dialogue state tracking. \n The following is the dialogue you need to test: {text} \n Please return the value of slot: <{domain_slot}>. It indicates {slot_explain}. {possible_values} If the slot is not mentioned in the dialogue, just return NONE.\n So the value of slot <{domain_slot}> is \n"
        possible_values = self.possible_values_prompt.format(
            ", ".join(self.slot_possible_values[f"{domain}-{slot}"])
        ) if f"{domain}-{slot}" in self.slot_possible_values else ""
        input_text = intput_template.format(
            text=text,
            domain=domain,
            domain_explain=self.domain_explain[domain],
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
            instruction=self.instruction,
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


class DatabasedPrompter:
    def __init__(self, data_path, schema_path):
        self.domain_explain = {}
        self.slot_explain = {}
        self.db = {}
        self.possible_values = {}

        for domain_entity in json.load(open(schema_path, "r")):
            self.domain_explain[domain_entity["service_name"]] = domain_entity["description"]
            for slot_entity in domain_entity["slots"]:
                self.slot_explain[slot_entity["name"]] = slot_entity["description"]
        # for filename in glob.glob(os.path.join(data_path, "*_db.json")):
        #     domain = os.path.basename(filename).split("_")[0]
        #     print(domain)
        #     self.db[domain] = json.load(open(filename, "r"))
        self.possible_values = json.load(open(os.path.join(data_path, "ontology.json"), "r"))
        # print(self.possible_values["taxi-leaveAt"])
        self.input_template_gpt = '''Perform the task of multi-domain dialogue state tracking.
The following is the dialogue you need to test: {text} 
Domain: {domain}, it indicates {domain_explain}. Slot: {slot}, it indicates {slot_explain}.
Please return the value of slot: <{domain}-{slot}>. You can only choose from the following available values: {possible_values}.
If the the slot or domain is not mentioned in the provided dialogue, just output "not mentioned".
If the user explicitly said he/she doesn’t care about the slot in the dialogue, just output "dontcare".
'''
        # Focus only on the values mentioned in the last utterance.
        # If the slot is mentioned but it is not the user's requirement or it has not received a positive response by system, also output "not mentioned".
        # First return whether the slot is mentioned in the dialogue or not, then write the value of slot. In the format of "mentioned: xxx; value: xxx".
        self.input_template_databased = '''I present you with some databases together with one example item and value constraints

Hotel
| name           | type        | parking | book_stay | book_day | book_people | area  | stars | internet | pricerange |
| hamilton lodge | guest house | no      |         1 | tuesday  |           6 | north |     5 | free     | cheap      |
area can be one of [lspeast, south, centre, dontcare, north, west
]
stars can be one digit from 0 to 5
parking can be one of [free, yes, dontcare, no
]
internet can be one of [free, yes, dontcare, no
]
book_day can be one day in a week
book_people can be one digital number
book_stay can be one number from 1 to 5
type can be 'guest house' or hotel
pricerange can be one of [expensive, cheap, moderate, dontcare
]


Train
| departure          | destination | arriveby | leaveat | book_people | day      |
| bishops stortford  | cambridge   |    20: 07 |   13: 32 |           2 | monday   |
day can be one day in a week
arriveby, leaveat can be a 'HH:MM' timestamp


Attraction
| name                    | type          | area     |
| pembroke college        | outdoor       | west     |
type can be one of [cinema, museum, multiple sports, dontcare, concert hall, pool, outdoor, church, park, entertainment, swimming pool, theatre, college, architecture, boat, nightclub
]
area can be one of [east, north, south, dontcare, centre, west
]


Restaurant
| name                    | food       | book_people | book_day | book_time | area     | pricerange |
| hotel du vin and bistro | chinese    |           2 | saturday |     18: 30 | dontcare | cheap      |
pricerange can be one of [expensive, cheap, moderate, dontcare
]
area can be one of [east, south, centre, dontcare, north, west
]


Taxi
| arriveby | leaveat | departure                        | destination                     |
|    12: 30 |   24: 30 | cambridge artworks               | warkworth house                 |


I need you to look at a series of conversation turns between system and user, and output all attributes requested by the user. I'll give you all turns one by one.
Your need to output in following form for every turn I give you:
User informed 'N' columns: <table name>-<column>=<value>; ...
For example:
User informed 2 columns: restaurant-food=chinese; hotel-area=dontcare
You must output attributes that is valid in above tables
**Important:** you should't output any attributes that have alrealy appeared in any previous lists.

Next is the first turn: 1th turn:
System: 
User: hi , can you give me some information on a place to stay in cambridge ? i would prefer some place expensive .

Please write the lists: (Don't write anything other than the lists themselves)'''

    def generate_message_history(self, text, domain, slot):
        domain_slot = domain + "-" + slot
        return [{
            "role": "user",
            "content": self.input_template_gpt.format(
                text=text,
                domain=domain,
                slot=slot,
                domain_explain=self.domain_explain[domain],
                slot_explain=self.slot_explain.get(domain_slot.lower().replace(" ", ""), ""),
                possible_values=self.possible_values[domain_slot]
            )
        }]

    def get_response(self, response: str) -> str:
        if " is:" in response:
            response = response.split(" is:")[1]
        if " is " in response:
            response = response.split(" is ")[1]
        response = response.split(",")[0].split(".")[0].replace("</s>", "").replace("<s>", "").replace("\"", "").strip()
        return response


class SingleReturnPrompter:
    def __init__(self):
        self.ontology = json.load(open("ontology.json"))
        self.prompt_template = """
I present you with some databases together with one example item and value constraints

Hotel
| name           | type        | parking | book_stay | book_day | book_people | area  | stars | internet | pricerange |
| hamilton lodge | guest house | no      |         1 | tuesday  |           6 | north |     5 | free     | cheap      |
area can be one of [east, south, centre, north, west]
stars can be one digit from 0 to 5
parking can be one of [free, yes, no]
internet can be one of [free, yes, no]
book_day can be one day in a week
book_people can be one digital number
book_stay can be one number from 1 to 5
type can be 'guest house' or hotel
pricerange can be one of [expensive, cheap, moderate]


Train
| departure          | destination | arriveby | leaveat | book_people | day      |
| bishops stortford  | cambridge   |    20:07 |   13:32 |           2 | monday   |
day can be one day in a week
arriveby, leaveat can be a 'HH:MM' timestamp


Attraction
| name                    | type          | area     |
| pembroke college        | outdoor       | west     |
type can be one of [cinema, museum, multiple sports, concert hall, pool, outdoor, church, park, entertainment, swimming pool, theatre, college, architecture, boat, nightclub]
area can be one of [east, north, south, centre, west]


Restaurant
| name                    | food       | book_people | book_day | book_time | area     | pricerange |
| hotel du vin and bistro | chinese    |           2 | saturday |     18:30 | dontcare | cheap      |
pricerange can be one of [expensive, cheap, moderate]
area can be one of [east, south, centre, north, west]


Taxi
| arriveby | leaveat | departure                        | destination                     |
|    12:30 |   24:30 | cambridge artworks               | warkworth house                 |


I need you to look at one conversation turn between system and user, where the system tries to find an item in the database for the user with all column constraints provided by the user. You need to output all attributes provided/informed by the user in the sentence I give you. I'll give you one turn and already determined dialogue contexts from previous dialogue turn.
Your need to output in following form for every turn I give you:
User informed 'N' columns: <table name>-<column>=<value>; <table name>-<column>=<value>; ...
For example:
User informed 2 columns: restaurant-food=chinese; hotel-area=east

You must output attributes that is valid in above tables
You should't output any attributes that have alrealy appeared in any previous lists.
If and only if user explicitly said he/she doesn't care about some attribute, you should output '<dontcare>' for that attribute.
If user asked a attribute from the system, you should output '<request>' for that attribute.
If user doesn't inform any attribute, you should output 'No column informed' for that turn.

Next is the dialogue turn and current context:
Contexts: {input_context}
Dialogue:
{input_utterance}
Please write the lists: (Don't write anything other than the lists themselves)
"""

    def generate_prompt(self, system_uttr, user_uttr, last_state):
        if system_uttr == "":
            context = "This is the first turn"
        else:
            context = "; ".join(
                [f"{k}={v}" for k, v in last_state.items()]
            )
        context = ""
        return self.prompt_template.format(
            input_context=context,
            input_utterance=f"sys: {system_uttr} \n usr: {user_uttr}"
        )

    def get_response(self, output):
        if "column" not in output and "informed" not in output:
            return {
                "FAILED": "PARSE FAILED"
            }
        output = output.split("\n\n")[0].replace("\n", " ")
        pat = re.findall(r"\d+ column(s)?: (.*)", output)
        if len(pat) == 0:
            return {}
        pat = pat[0]
        sv_str = pat[1]
        if ";" in sv_str:
            sv_str = sv_str.split(";")
        elif "*" in sv_str:
            sv_str = sv_str.split("*")
        preds = {}
        for sv in sv_str:
            if "=" not in sv or len(sv.split("=")) != 2:
                continue
            slot, value = sv.split("=")
            slot = slot.strip("<> ").lower()
            value = value.strip(".<> ").lower()
            preds[slot] = value
        preds = self.typo_fix(preds)
        return preds

    def typo_fix(self, preds, version="2.4"):

        # fix the named entities in these slots
        named_entity_slots = ['hotel-name', 'train-destination', 'train-departure',
                              'attraction-type', 'attraction-name',
                              'restaurant-name', 'taxi-departure', 'taxi-destination', 'restaurant-food']
        fixed = {}
        for slot, value in preds.items():
            # _ in slot should be ' '
            # slot = slot.replace('_', ' ')

            # fix 's
            value = value.replace(' s ', 's ')
            if value.endswith(' s'):
                value = value[:-2] + 's'

            # remove "
            value = value.replace('"', '')

            # fix typo words
            general_typos = {'fen ditton': 'fenditton',
                             'guesthouse': 'guest house',
                             'steveage': 'stevenage',
                             'stantsted': 'stansted',
                             'storthford': 'stortford',
                             'shortford': 'stortford',
                             'weish': 'welsh',
                             'bringham': 'birmingham',
                             'liverpoool': 'liverpool',
                             'petersborough': 'peterborough',
                             'el shaddai': 'el shaddia',
                             'wendesday': 'wednesday',
                             'brazliian': 'brazilian',
                             'graffton': 'grafton'}
            for k, v in general_typos.items():
                value = value.replace(k, v)

            # fix whole value
            value_replacement = {'center': 'centre',
                                 'caffe uno': 'cafe uno',
                                 'caffee uno': 'cafe uno',
                                 'christs college': 'christ college',
                                 'churchill college': 'churchills college',
                                 'sat': 'saturday',
                                 'saint johns chop shop house': 'saint johns chop house',
                                 'good luck chinese food takeaway': 'good luck',
                                 'asian': 'aspian oriental',
                                 'gallery at 12': 'gallery at 12 a high street'}

            if version == "2.1":
                value_replacement['portuguese'] = 'portugese'
                value_replacement['museum of archaeology and anthropology'] = 'museum of archaelogy and anthropology'

            if version == "2.4":
                value_replacement['portugese'] = 'portuguese'
                value_replacement['museum of archaelogy and anthropology'] = 'museum of archaeology and anthropology'

            for k, v in value_replacement.items():
                if value == k:
                    value = v

            # time format fix `after 17:00` -> 17:00
            if ':' in value:
                time_stamps = re.findall(r'\d{2}:\d{2}', value)
                if len(time_stamps) > 0:
                    value = time_stamps[0]

            if slot in named_entity_slots:
                value = self.check_prefix_suffix(value, self.ontology[slot])

            # skip slots where value/slot contains 'request'
            if 'request' in value or 'request' in slot:
                continue

            if value:
                fixed[slot] = value.strip()
        return fixed

    def check_prefix_suffix(self, value, candidates):
        # add/delete "the" in the front, or the suffix in the end.
        if value in candidates:
            return value
        prefixes = ['the ']
        suffixes = [" hotel", " restaurant", ' cinema', ' guest house',
                    " theatre", " airport", " street", ' gallery', ' museum']
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        for suffix in suffixes:
            if value.endswith(suffix):
                value = value[:-len(suffix)]
                break
        for prefix in [''] + prefixes:
            for suffix in [''] + suffixes:
                possible_value = prefix + value + suffix
                if possible_value in candidates:
                    return possible_value
        return value

