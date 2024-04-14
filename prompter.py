import json
import re


class SingleReturnPrompter:
    def __init__(self):
        self.ontology = json.load(open("ontology.json"))
        self.short_prompt_template = """Contexts: {input_context}
Dialogue:
{input_utterance}
Please write the lists: (Don't write anything other than the lists themselves)
"""
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
        # context = ""
        return self.short_prompt_template.format(
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
        else:
            sv_str = [sv_str]
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

