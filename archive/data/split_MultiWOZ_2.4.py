import glob
import json
import os.path

data_path = "MULTIWOZ2.4"
save_path = "MULTIWOZ2.4/split"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# slots_under_domain = {}
# schema = json.load(open(os.path.join(data_path, "ontology.json")))
# for domain_slot in schema:
#     domain, slot = domain_slot.split("-")
#     if domain not in slots_under_domain:
#         slots_under_domain[domain] = []
#     slots_under_domain[domain].append(slot)

data = json.load(open(os.path.join(data_path, "data.json"), "r"))
val_indexes = set(line.strip() for line in open(os.path.join(data_path, "valListFile.json"), "r").readlines())
test_indexes = set(line.strip() for line in open(os.path.join(data_path, "testListFile.json"), "r").readlines())
train_data = {}
test_data = {}
val_data = {}
for index, content in data.items():
    if index in val_indexes:
        val_data[index] = content
    elif index in test_indexes:
        test_data[index] = content
    else:
        train_data[index] = content
json.dump(train_data, open(os.path.join(save_path, "train.json"), "w"))
json.dump(test_data, open(os.path.join(save_path, "test.json"), "w"))
json.dump(val_data, open(os.path.join(save_path, "val.json"), "w"))
