## MultiWOZ24_preprocess.py

### Inputs

`./MULTIWOZ2.4/*` as MultiWOZ_2.4 format
`./MultiWOZ_2.1/slot_descriptions.json`

### Outputs

`output_path = ./MULTIWOZ2.4_preprocess/`

`$output_path/train.json`

json lines, for every line:
```
{   
    # for every turn every domain every slot
    "dialogue": "[USER] xxx [SYSTEM] xxx ... [domain] _domain_, [slot] _slot_, it indicates _slot_description_, [Possible Values] xxx, xxx ... ",
    "state":  _value_|"NONE" 
}
```

`$output_path/train.idx`

lines, for every line:
```
data.json|||_index_|||_turn_|||_domain_id_|||_domain_|||_slot_
# for MultiWOZ_2.2 data.json will be dialogue_xxx.json
# _domain_id_ is the order in ontology.json starts from 0
```
xxx.json xxx.idx is in the same order

`$output_path/test.json` same as above

`$output_path/test.idx` same as above

`$output_path/val.json` same as above

`$output_path/val.idx` same as above

## data_prepare_zero-shot_MultiWOZ24.py

### inputs

`./MULTIWOZ2.4_preprocess/train.json`

`./MULTIWOZ2.4_preprocess/train.idx`

### outputs

`./MULTIWOZ2.4_preprocess/train_LLM_zero-shot_except-attraction-domain.json`

activate_number = 475217 # 建议搬运 data_prepare_few-shot_MultiWOZ20.py 中的 Count_activate_state_number

none_sample_number = all - activate_number

```
[{
    "instruction": shown in figure 4,
    "input": shown in figure 4,
    "output": _value_
}]
```

随机去掉一些 state 为 NONE 的对话使得 NONE 与 非NONE 数量大致相等，这一步使得 `.idx` 不能再使用

## generate_zero-shot.py

### inputs

`./xxxx/test_LLM_xxxx.json` same as above

`./xxxx/test_LLM_xxxx.idx` idx for xxx_LLM_xxx.json but **no script generate it**

### outputs

`./xxxx/test_LLM_result.txt` if exists continue instead of start from 0 

every line:
```
xxx.json|||_index_|||_turn_|||_domain_id_|||_domain_|||_slot_|||[_response_]
# _response_ is valid only _domain_==except_domain
```
notice that every idx description copied from test_LLM_xxxx.idx

## postprocess.py

### inputs

`$prediction_txt` format described above

`$data_dir/test/dialogues_001.json` in MultiWOZ_2.2 format

`$data_dir/test/dialogues_002.json` same as above

### outputs

`$out_dir/dummy_out_dialogue_001.json`

```
[{
    "dialogue_id": _index_,
    "turns":[{
        "turn_id": _turn_,
        "speaker": "USER"|"SYSTEM",
        "frames": [{ # USER only
            "service": _domain_,
            "state": {
                "slot_values":{ # leave blank if $prediction_txt does not contain this sample
                    _domain_-_slot_: [ _value_ ] # for not NONE predicted values
                }
            }]
        },
        "utterance": str # SYSTEM only
    }]
}]
```

`$out_dir/dummy_out_dialogue_002.json` same as above


## eval.py

### inputs

`$data_dir/$eval_set/dialogue_*.json`

`$prediction_dir/*.json` # dummy_out_dialogue_xxx.json

``

### outputs

`$output_metric_file`
```
{
    _domain_: {
        "average_goal_accuracy": float|"NA",
        "average_cat_accuracy": float|"NA",
        "average_noncat_accuracy": float|"NA",
        "joint_goal_accuracy": float|"NA",
        "joint_cat_accuracy": float|"NA",
        "joint_noncat_accuracy": float|"NA"
    }
}
```

`$prediction_dir/dialogues_and_metrics.json`

```
{
    _index_: { ... } # MultiWOZ_2.2 format data entity
}
```