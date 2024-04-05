import json
import os

preprocessed_path = "MultiWOZ_2.4_preprocess"
predicted_path = "MultiWOZ_2.4_preprocess"

for data_type in ["test"]:
    ground_truth = json.load(open(os.path.join(preprocessed_path, data_type + ".json")))
    predicted = json.load(open(os.path.join(predicted_path, data_type + "_out_LDST_10p.json")))
    assert len(ground_truth) == len(predicted)
    AGA_score = 0
    ground_truth_index = {}
    for sample in ground_truth:
        ground_truth_index[f'{sample["index"]}|{sample["turn"]}|{sample["domain"]}|{sample["slot"]}'] = sample["value"].replace("not mentioned", "none")
    predicted_index = {}
    for sample in predicted:
        Y = ground_truth_index[f'{sample["index"]}|{sample["turn"]}|{sample["domain"]}|{sample["slot"]}']
        if Y.lower() == sample["value"].replace("not mentioned", "none").lower():
            AGA_score += 1
        # predicted_index[f'{sample["index"]}|{sample["turn"]}|{sample["domain"]}|{sample["slot"]}'] = sample["value"]
    # ground_truth_list = [ground_truth_index[key] for key in sorted(ground_truth_index.keys())]
    # predicted_list = [predicted_index[key] for key in sorted(predicted_index.keys())]
    # for truth, pred in zip(ground_truth_list, predicted_list):
    #     if truth == pred:
    #         AGA_score += 1
    AGA_score /= len(ground_truth)

    JGA_score = 0
    ground_truth_index = {}
    for sample in ground_truth:
        ground_truth_index[f'{sample["index"]}|{sample["turn"]}'] = \
            ground_truth_index.get(f'{sample["index"]}|{sample["turn"]}', "") + \
            f'|{sample["domain"]}|{sample["slot"]}|{sample["value"]}'
    predicted_index = {}
    for sample in predicted:
        predicted_index[f'{sample["index"]}|{sample["turn"]}'] = \
            predicted_index.get(f'{sample["index"]}|{sample["turn"]}', "") + \
            f'|{sample["domain"]}|{sample["slot"]}|{sample["value"]}'
    ground_truth_list = [ground_truth_index[key] for key in sorted(ground_truth_index.keys())]
    predicted_list = [predicted_index[key] for key in sorted(predicted_index.keys())]
    for truth, pred in zip(ground_truth_list, predicted_list):
        if truth == pred:
            JGA_score += 1
    JGA_score /= len(ground_truth_list)

    print("Average Goal Accuracy:", AGA_score)
    print("Joint Goal Accuracy:", JGA_score)
    print("average slots per frame:",  len(ground_truth) / len(ground_truth_list))
