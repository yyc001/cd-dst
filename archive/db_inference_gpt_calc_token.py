import json
import os.path

import tiktoken
from openai import OpenAI

first_turn_prompt = """I present you with some databases together with one example item and value constraints

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

Next is the first turn:
{turn_text}

Please write the lists: (Don't write anything other than the lists themselves)"""

turn_prompt = '''
{turn_id}th turn:
{turn_text}
'''

data_path = "data/MULTIWOZ2.4"

tot_token = 0
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
for data_type in ["test"]:
    data = json.load(open(os.path.join(data_path, "split", data_type + ".json")))
    for index, dialogue in data.items():
        turn_text = ""
        messages = []
        predicted_turn_state = {}
        for turn_id, turn in enumerate(dialogue["log"]):
            print("turn", turn_id, "--------------------------")
            speaker = ["User", "System"][turn_id % 2]
            if speaker == "User":
                turn_text = ""
            else:
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
                response = "k "*10
                messages.append({
                    "role": "assistant",
                    "content": response
                })
            turn_text += f"{speaker}: {turn['text']}\n"
