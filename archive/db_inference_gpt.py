import json
import os.path

from openai import OpenAI

first_turn_prompt = """I present you with some databases together with one example item and value constraints

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

Next is the first turn:
{turn_text}

Please write the lists: (Don't write anything other than the lists themselves)"""

turn_prompt = '''
{turn_id}th turn:
{turn_text}
'''

data_path = "data/MULTIWOZ2.4"

ontology = json.load(open(os.path.join(data_path, "../ontology.json")))
domains = set(key.split("-")[0] for key in ontology)
client = OpenAI(
    api_key="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"
)


def extract_state(response):
    predicted_turn_state = {}
    try:
        extracted_state = [ans.split("=") for ans in response.split(": ")[1].split(";")]
        for iii in extracted_state:
            if len(iii) != 2:
                continue
            slot, value = iii
            predicted_turn_state[slot.strip().lower()] = value.strip().lower()
    except IndexError:
        print(response)
    return predicted_turn_state


def compare_state(a, b):
    if len(a) != len(b):
        return 0
    for slot, value in a.items():
        if slot not in b or value != b[slot]:
            return 0
    return 1

token_used = 0
for data_type in ["test"]:
    data = json.load(open(os.path.join(data_path, "split", data_type + ".json")))
    tot_turn = 0
    acc_turn = 0
    for index, dialogue in data.items():
        turn_text = ""
        turn_state = {}
        messages = []
        predicted_turn_state = {}
        for turn_id, turn in enumerate(dialogue["log"]):
            print("turn", turn_id, "--------------------------")
            speaker = ["User", "System"][turn_id % 2]
            if speaker == "User":
                turn_text = ""
                turn_state = {}
            else:
                for domain in turn["metadata"]:
                    for slot, value in turn["metadata"][domain]["semi"].items():
                        if value not in ["", "none", "not mentioned"]:
                            turn_state[f"{domain}-{slot}".lower()] = value
                    for slot, value in turn["metadata"][domain]["book"].items():
                        if slot != "booked" and value not in ["", "none", "not mentioned"]:
                            turn_state[f"{domain}-book_{slot}".lower()] = value
                print(turn_state)
                prompt = turn_prompt.format(
                    turn_id=turn_id // 2 + 1,
                    turn_text=turn_text
                )
                if turn_id == 1:
                    prompt = first_turn_prompt.format(turn_text=prompt)
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                token_used += response.usage.total_tokens
                response = response.choices[0].message.content
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                predicted_turn_state.update(**extract_state(response))
                for slot, value in turn_state.items():
                    if slot not in predicted_turn_state:
                        print(slot, ":", value, "->", "none")
                    elif value != predicted_turn_state[slot]:
                        print(slot, ":", value, "->", predicted_turn_state[slot])
                for slot, value in predicted_turn_state.items():
                    if slot not in turn_state:
                        print("slot", slot, ":", "none", "->", value)
                tot_turn += 1
                acc_turn += compare_state(turn_state, predicted_turn_state)
                print("JGA=", acc_turn/tot_turn, "token used:", token_used)
            turn_text += f"{speaker}: {turn['text']}\n"
