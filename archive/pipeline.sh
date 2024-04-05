conda create -n -y py37 python=3.7
conda activate py37
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers accelerate bitsandbytes scikit-learn peft sentencepiece

cd data
git clone https://github.com/budzianowski/multiwoz.git
python preprocess_MultiWOZ_2.2.py
cd ..

git clone https://github.com/Elegybackup/clash-for-linux-backup.git
cd clash-for-linux-backup
vim .env
bash start.sh

python finetune.py > finetune.out 2> finetune.err

shutdown