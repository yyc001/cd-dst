``` 
python main.py --job=inference --model_config=model/flan-t5-xxl.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/flan_t5_xxl.json --resume
python main.py --job=evaluation --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/llama-2.json --reparse
python main.py --job=inference --model_config=model/tuned-llama.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/tuned-llama.json --resume
python main.py --job=inference --model_config=model/contra-t5.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/contra-t5-results.json --resume

python main.py --job=inference --model_config=model/flan-t5-xxl.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=flan_t5_xxl-test-results.json --resume
python main.py --job=evaluation --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/tuned-llama.json --reparse
python main.py --job=evaluation --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/tuned-t5.json --reparse
python main.py --job=inference --model_config=model/mistral.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_processed/mistral-tuned.json --resume

python main.py --job=inference --model_config=model/tuned-llama.json --data_path=data/MultiWOZ_2.4_processed/test.json --output_file=data/MultiWOZ_2.4_output/tuned-llama.json --resume
