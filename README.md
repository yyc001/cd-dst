``` 
python main.py --job=inference --model_config=model/flan-t5-xxl.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/flan_t5_xxl.json --resume
python main.py --job=evaluation --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/llama-2.json
python main.py --job=inference --model_config=model/tuned-llama.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/tuned-llama.json --resume

```