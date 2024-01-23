srun --gres=gpu:4 \
     --nodelist=gpu[08] \
     --job-name=lmft \
     --cpus-per-task=2 \
     python -m torch.distributed.launch --nproc_per_node=4 train.py