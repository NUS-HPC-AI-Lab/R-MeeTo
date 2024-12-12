export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_PATH="./datasets/imagenet/"
MODEL_PTH="./pretrained"

# has order
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29459 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --epoch 3 --distill True
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29459 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_small --merge_method ToMe --num_prune 11 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --epoch 3 --distill True
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29459 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_base --merge_method ToMe --num_prune 11 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --epoch 3 --distill True
# no order
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29459 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric X --model_pth $MODEL_PTH --batch-size 128 --distance cosine --epoch 3 --distill True
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29459 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_small --merge_method ToMe --num_prune 11 --metric X --model_pth $MODEL_PTH --batch-size 128 --distance cosine --epoch 3 --distill True
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29459 main.py --lr 2e-5 --min-lr 1e-6 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_base --merge_method ToMe --num_prune 11 --metric X --model_pth $MODEL_PTH --batch-size 128 --distance cosine --epoch 3 --distill True