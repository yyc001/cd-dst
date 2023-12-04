import glob
import json
import os.path

data_path = "multiwoz/data/MultiWOZ_2.2"
predicted_file = "MultiWOZ_2.2_preprocess_with_none"
save_path = "MultiWOZ_2.2_postprocess"
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

for data_type in ["test"]:
    predicted_list = json.load(open(os.path.join(predicted_file, data_type + "_out.json"), "r"))
    pred_index = {}
    for prediction in predicted_list:
        if prediction["value"] != "none":
            index = f'{prediction["index"]}|{prediction["turn"]}|{prediction["domain"]}|{prediction["slot"]}'
            pred_index[index] = prediction["value"]
    filenames = glob.glob(os.path.join(data_path, data_type, "dialogues_*.json"))
    for filename in filenames:
        data_ori = json.load(open(filename, "r"))
        basename = os.path.basename(filename)
        pred_entity = [{
            "dialogue_id": dialog["dialogue_id"],
            # "services": [ _domain_ ],
            "turns": [{
                "frames": [{  # SYSTEM's turn if frames empty else 8
                    # "actions": [],
                    "service": frame["service"],
                    # "slots": [{
                    #     "slot": _domain_-_slot_,
                    #     "value": _value_,
                    #     "exclusive_end": int,
                    #     "start": int
                    # }],
                    "state": {
                        # "active_intent": _intent_, # e.g. find_train,
                        # "requested_slots": [],
                        "slot_values": {
                            slot: [pred_index[f'{dialog["dialogue_id"]}|{turn["turn_id"]}|{frame["service"]}|{slot}']]
                            for slot in slots_under_domain[frame["service"]]
                            if f'{dialog["dialogue_id"]}|{turn["turn_id"]}|{frame["service"]}|{slot}' in pred_index
                        }
                    }
                } for frame in turn["frames"]],
                "speaker": turn["speaker"],
                "turn_id": turn["turn_id"],
                # "utterance": str # dialog content
            } for turn in dialog["turns"]]
        } for dialog in data_ori]
        if not os.path.exists(os.path.join(save_path, data_type)):
            os.mkdir(os.path.join(save_path, data_type))
        save_file = open(os.path.join(save_path, data_type, basename), "w")
        json.dump(pred_entity, save_file)
