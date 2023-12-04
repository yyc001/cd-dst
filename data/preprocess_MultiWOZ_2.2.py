import glob
import json
import os.path

data_path = "multiwoz/data/MultiWOZ_2.2"
save_path = "MultiWOZ_2.2_preprocess"
if not os.path.exists(save_path):
    os.mkdir(save_path)

slots_under_domain = {}
schema = json.load(open(os.path.join(data_path, "schema.json")))
for domain_entity in schema:
    domain = domain_entity["service_name"]
    slots_under_domain[domain] = []
    for slot_entity in domain_entity["slots"]:
        slot = slot_entity["name"]
        slots_under_domain[domain].append(slot)

for data_type in ["train", "dev", "test"]:
    filenames = glob.glob(os.path.join(data_path, data_type, "dialogues_*.json"))
    data_ori = []
    for filename in filenames:
        data_ori.extend(json.load(open(filename, "r")))
    output_samples = []
    for sample in data_ori:
        index = sample["dialogue_id"]
        domains = sample["services"]
        dialogue_history = ""
        for turn_id, turn in enumerate(sample["turns"]):
            dialogue_history += f"[{turn['speaker']}] {turn['utterance'] }"
            if turn_id % 2 == 1:
                continue
            this_turn_samples = {}
            for frame in turn["frames"]:
                domain = frame["service"]
                if domain in domains and "state" in frame:
                    slots = frame["state"]["slot_values"]
                    for slot, value in slots.items():
                        this_turn_samples[slot] = {
                            "index": index,
                            "turn": turn_id,
                            "dialogue": dialogue_history,
                            "domain": domain,
                            "slot": slot,
                            "value": value[0]
                        }
            for domain in domains:
                for slot in slots_under_domain[domain]:
                    if slot not in this_turn_samples:
                        this_turn_samples[slot] = {
                            "index": index,
                            "turn": turn_id,
                            "dialogue": dialogue_history,
                            "domain": domain,
                            "slot": slot,
                            "value": "none"
                        }
            output_samples.extend(list(this_turn_samples.values()))

    output_file = open(os.path.join(save_path, data_type + ".json"), "w")
    json.dump(output_samples, output_file)


