### 取值
```
_index_ : ["MUL2700.json", "PMUL4332.json", "SNG02342.json", "SSNG0388.json", "WOZ20676.json", ...]
_domain_: ["attraction", "hospital", "hotel", "police", "restaurant", "taxi", "train", "bus"]
_slot_: described in ontology.json 
_value_: described in ontology.json
_entity_: [['info', 'reqt', 'fail_info'], ['info', 'fail_info', 'book', 'fail_book'], ['info', 'fail_info', 'hotel', 'book', 'fail_book'], ['info', 'reqt', 'fail_info', 'fail_book'], ['info', 'restaurant', 'fail_info', 'hotel', 'book', 'fail_book'], ['info', 'fail_info', 'book']]
```
对于 `_index_` , "MUL" means 2-3 domains, "PMUL" means 1-3 domain, else means 1 domain

 ### ontology.json
```
{
    _domain_-_slot_: [ _value_ ]
}
```

### data.json

```
{
    _index_: {
        "goal": {
            "message":[str], # a list of HTML text, means the task user to recieve 
            _domain_: { # domains except bus
                # info(book) 和 fail_info(book) 很迷，有时fail没在message里出现
                *"info": { _slot_: _value_, *"invalid":false ... }, # user's final?(second?) choice???
                *"fail_info": { _slot_: _value_, "*pre_invalid":false, ... }, # user's first choice but failed???
                *"book": { _slot_: _value_, ... }, # same as above
                *"fail_book": { _slot_: _value_, ... }, # same as above
                *"reqt": [ ... ],
                *"hotel": str|false, # SSNG only
                *"restaurant": str|false # SSNG only
            },
            *"topic": { _domain_: false, "booking": false, "general": false } # all false
        }, 
        "log":[{
            "text": str, # user,agent,user,agent.... dialogue
            "metadata": { # user's turn if metadata is empty
                _domain_: {
                    "book": {
                        "booked": [{...}], # the ID flag user should acquire, like trainID, reference number etc.
                        _slot_: _value_ # "_domain_-book _slot_" described in ontology.json
                    },
                    "semi": { _slot_: _value_ } # "_domain_-_slot_" described in ontology.json
                }
            },
            *"dialog_act": { ... }, # MultiWOZ_2.1 only
            *"span_info": { ... }, # MultiWOZ_2.1 only
        }]
    }
}
```

### valListFile.json & testListFile.json (.txt for MultiWOZ_2.1)

each file 1000 lines, each line contains a `_index_` indicates this data should be split to val|test data?

### dialogue_acts.json (system_acts.json for MultiWOZ_2.1)
```
{
    _index_: { # no ".json" suffix
        _turn_/2(?): { ... } # some key information each turn agent should? say
    } 
}
```

### slot_descriptions.json (MultiWOZ_2.1 only)
```
{
    _domain_-_slot_: [str] # at least 2 descriptions
}
```

### schema.json (MultiWOZ_2.2 only)
可能是集合了 2.1 的 `slot_descriptions.json` 和 `ontology.json`
```
[
    {
        "service_name": _domain_,
        "slots": [{
            "name": _domain_-_slot_, # no space in book slots
            "description": str,
            "possible_values": [ _value_ ], # empty if is_categorical is false
            "is_categorical": bool
        }],
        "intents": [{
            "name": _intent_,
            "description": "search for places to see for leisure",
            "is_transactional": bool,
            "required_slots": [],
            "optional_slots": { _domain_-_slots_: "dontcare"}
        }]
    }
]
```

### dialogue_xxx.json (MultiWOZ_2.2 only)
```
[{
    "dialogue_id": _index_,
    "services": [ _domain_ ],
    "turns": [{
        "frames": [{ # SYSTEM's turn if frames empty else 8
            "actions": [],
            "service": _domain_,
            "slots": [{
                "slot": _domain_-_slot_,
                "value": _value_,
                "exclusive_end": int,
                "start": int
            }],
            "state": {
                "active_intent": _intent_, # e.g. find_train,
                "requested_slots": [], 
                "slot_values": { _domain_-_slot_: [ _value_ ]} # no space
            }
        }],
        "speaker": "USER"|"SYSTEM",
        "turn_id": _turn_,
        "utterance": str # dialog content
    }]
}]
```