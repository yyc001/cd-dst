import json
from tqdm import tqdm


def main(
        processed_data_path="data/MultiWOZ_2.2_preprocess/test.json",
        output_file="data/MultiWOZ_2.2_preprocess/test_out_gpt.json"
):
    ground_truth = json.load(open(processed_data_path, "r"))
    # predicted = json.load(open(output_file, "r"))
    predicted = []
    for line in open(output_file, "r"):
        predicted.append(json.loads(line))
    aga_num = 0

    jga_num = 0
    jga_tot = -1
    last_index_turn = ""
    last_full_state = True
    tot = 0
    for ans, out in tqdm(zip(ground_truth, predicted)):
        # if ans['value'] == "none":
        #     continue
        tot += 1
        assert out['index'] == ans['index']
        assert out['turn'] == ans['turn']
        assert out['domain'] == ans['domain']
        assert out['slot'] == ans['slot']

        this_index_turn = f'{ans["index"]}|{ans["turn"]}'
        if last_index_turn == "":
            last_index_turn = this_index_turn
        if last_index_turn != this_index_turn:
            last_index_turn = this_index_turn
            jga_tot += 1
            if last_full_state:
                jga_num += 1
            last_full_state = True

        if ans['value'].lower().replace("none", "not mentioned").replace(" ", "") == \
                out['value'].lower().replace("none", "not mentioned").replace(" ", ""):
            aga_num += 1
            if ans["value"] == "dontcare":
                print(ans["dialogue"])
                print(ans["domain"], "|||", ans["slot"], "|||", ans["value"], "|||", out['value'])
        else:
            last_full_state = False
            print(ans["dialogue"])
            print(ans["domain"], "|||", ans["slot"], "|||", ans["value"], "|||", out['value'])
    print("AGA:", aga_num / tot)
    print("JGA:", jga_num / jga_tot)


if __name__ == "__main__":
    main()
