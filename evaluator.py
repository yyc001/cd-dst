import json
import logging


class Evaluator:
    def __init__(self):
        ontology = json.load(open("ontology.json"))
        self.slot_space = set(key.lower().replace(" ", "_") for key in ontology.keys())
        logging.debug(self.slot_space)

    def evaluate(self, predicted, data, pred_only):
        if pred_only:
            data = {k: data[k] for k in predicted}

        turn_num = sum(len(dialogue) for dialogue in data.values())
        sv_num = sum(len(dialogue[-1]["state"]) for dialogue in data.values())
        predicted_sv_num = 0
        correct_turn_num = 0
        correct_sv_num = 0
        strict_hallucination_num = 0
        soft_hallucination_num = 0
        correct_svb_num = 0

        for index, dialogue in data.items():
            context = ""
            # print("!!!!!!!!!!!")
            for i in range(len(dialogue)):
                # print("#############", dialogue[i]["active_state"], predicted[index][i]["active_state"], dialogue[i]["state"])

                context += " " + dialogue[i]["system_utterance"] + " " + dialogue[i]["user_utterance"]
                result = self.compare_state(dialogue[i]["state"], predicted[index][i]["state"], context)
                correct_turn_num += result["correct_turn_num"]
                correct_sv_num += result["correct_sv_num"]
                correct_svb_num += result["correct_svb_num"]
                strict_hallucination_num += result["strict_hallucination_num"]
                soft_hallucination_num += result["soft_hallucination_num"]
                predicted_sv_num += result["predicted_sv_num"]

        return {
            "Num of dialogue": len(data),
            "Num of turn": turn_num,
            "Num of slot-value pair": sv_num,
            "Num of predicted slot-value pair": predicted_sv_num,
            "Joint Goal Accuracy": correct_turn_num / turn_num,
            "Active Slot Accuracy": correct_sv_num / sv_num,
            "Total Slot Accuracy": correct_svb_num / turn_num / 37,
            "Strict Hallucination Rate": strict_hallucination_num / predicted_sv_num / 2,
            "Soft Hallucination Rate": soft_hallucination_num / predicted_sv_num / 2
        }

    def compare_state(self, ground_truth, predicted, context):
        context = context.replace("\n", " ").lower()
        if "FAILED" in predicted:
            return {
                "correct_turn_num": 0,
                "correct_sv_num": 0,
                "strict_hallucination_num": 2 * len(ground_truth),
                "soft_hallucination_num": 0,
                "predicted_sv_num": len(ground_truth),
                "correct_svb_num": 37 - len(ground_truth)
            }

        joint = 1
        correct_sv = 0
        err_sv = 0
        for slot, value in predicted.items():
            if slot not in ground_truth:
                joint = 0
                err_sv += 1
        for slot, value in ground_truth.items():
            if slot in predicted and predicted[slot] == value:
                correct_sv += 1
            else:
                joint = 0
                err_sv += 1

        strict_hall = 0
        soft_hall = 0
        for slot, value in predicted.items():
            if slot not in self.slot_space:
                strict_hall += 1
            elif slot not in ground_truth:
                soft_hall += 1
            if value not in ground_truth.values():
                typo = {
                    "guest house": "guesthouse",
                    "churchills college": "churchill college"
                }
                if value in typo:
                    value = typo[value]
                if value not in ["dontcare", "yes", "no"] and value not in context:
                    logging.debug(f"+ {slot} | {value} | {ground_truth[slot] if slot in ground_truth else None} | {context}")
                    strict_hall += 1
                else:
                    soft_hall += 1

        if joint == 0:
            logging.debug(f"ERROR CASE --------------")
            logging.debug(f"ground_truth: {ground_truth}")
            logging.debug(f"predicted: {predicted}")

        return {
            "correct_turn_num": joint,
            "correct_sv_num": correct_sv,
            "strict_hallucination_num": strict_hall,
            "soft_hallucination_num": soft_hall,
            "predicted_sv_num": len(predicted),
            "correct_svb_num": 37 - err_sv
        }
