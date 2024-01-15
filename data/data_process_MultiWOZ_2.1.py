import glob
import json
import os.path

data_path = "MultiWOZ2.1"
save_path = "MultiWOZ_2.1_processed"
if not os.path.exists(save_path):
    os.mkdir(save_path)

possible_domains = ("restaurant", "attraction", "hospital", "police", "hotel", "train", "taxi")

data = json.load(open(os.path.join(data_path, "data.json"), "r"))
val_indexes = set(line.strip() for line in open(os.path.join(data_path, "valListFile.json"), "r").readlines())
test_indexes = set(line.strip() for line in open(os.path.join(data_path, "testListFile.json"), "r").readlines())
train_data = {}
test_data = {}
val_data = {}
for index, sample in data.items():
    last_user_utterance = ""
    last_system_utterance = ""
    last_state = {}
    dialogue = []
    for turn_id, turn in enumerate(sample["log"]):
        if len(turn["metadata"]) == 0:
            last_user_utterance = turn['text'].strip()
        else:
            last_system_utterance = turn['text'].strip()
            state = {}
            for domain in turn["metadata"]:
                if domain not in possible_domains:
                    continue
                for slot, value in turn["metadata"][domain]["semi"].items():
                    if len(value) > 0 and value != "not mentioned" and value.lower() != "none":
                        state[f"{domain}-{slot}".lower()] = value
                for slot, value in turn["metadata"][domain]["book"].items():
                    if slot == "booked":
                        continue
                    if len(value) > 0 and value != "not mentioned" and value.lower() != "none":
                        state[f"{domain}-book_{slot}".lower()] = value
            active_state = {k: v for k, v in state.items() if k not in last_state or last_state[k] != v}
            dialogue.append({
                "turn_id": turn_id // 2 + 1,
                "user_utterance": last_user_utterance,
                "system_utterance": last_system_utterance,
                "state": state,
                "active_state": active_state
            })
            last_state = state
    if index in val_indexes:
        val_data[index] = dialogue
    elif index in test_indexes:
        test_data[index] = dialogue
    else:
        train_data[index] = dialogue
json.dump(train_data, open(os.path.join(save_path, "train.json"), "w"), indent=2)
json.dump(test_data, open(os.path.join(save_path, "test.json"), "w"), indent=2)
json.dump(val_data, open(os.path.join(save_path, "val.json"), "w"), indent=2)
