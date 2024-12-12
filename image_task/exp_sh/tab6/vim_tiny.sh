export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_PATH="./datasets/imagenet/"
MODEL_PTH="./pretrained"

# eval
# metric X
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric X  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine --eval
# metric B
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric B  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine --eval
# metric C
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric C  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine --eval
# metric delta
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric delta  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine --eval

# retrain
# metric X
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric X  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine
# metric B
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric B  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine
# metric C
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric C  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine
# metric delta
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric delta  --if_order --model_pth $MODEL_PTH --batch-size 128 --distill True --distance cosine
